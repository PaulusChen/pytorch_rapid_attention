#include "flash_attention_operator.hpp"


#include <sys/types.h>
#include <type_traits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "common.h"

namespace rapid_flash_attention {

FlashForwardKernelConfig py_to_cpp_kernel_config(const py::object &py_cfg) {
  return FlashForwardKernelConfig{
      py::cast<torch::ScalarType>(
          py_cfg.attr("dtype").attr("to_torch_dtype")()),
      py::cast<int>(py_cfg.attr("d_head")),
      py::cast<int>(py_cfg.attr("B_r")),
      py::cast<int>(py_cfg.attr("B_c")),
      py::cast<int>(py_cfg.attr("n_warps")),
      py::cast<bool>(py_cfg.attr("async_copy")),
      py::cast<bool>(py_cfg.attr("eager_load_blocks")),
      py::cast<bool>(py_cfg.attr("swizzled")),
      py::cast<int>(py_cfg.attr("Q_mma_load_K_tiles")),
      py::cast<int>(py_cfg.attr("K_mma_load_K_tiles")),
      py::cast<int>(py_cfg.attr("V_mma_load_K_tiles")),
      py::cast<bool>(py_cfg.attr("mma_double_buffer_loads")),
      py::cast<bool>(py_cfg.attr("optimized_softmax"))};
}

/* @brief python绑定的Flash Attention前向函数
 *
 * @param py_cfg Python端传入的内核配置对象
 * @param TQ 查询张量，形状为 (batch_size, seq_len, n_heads, d_head)
 * @param TK 键张量，形状与TQ相同
 * @param TV 值张量，形状与TQ相同
 * @param out_ 可选的输出张量，如果提供则结果写入该张量，否则创建新的输出张量
 * @param benchmark 是否进行基准测试，返回运行时间
 *
 * @return 输出张量和运行时间（如果benchmark为true）
 */
std::tuple<at::Tensor, float>
flash_attention_forward(const py::object &py_cfg, const torch::Tensor &TQ,
                        const torch::Tensor &TK, const torch::Tensor &TV,
                        std::optional<at::Tensor> &out_, bool benchmark) {
  CHECK_INPUT(TQ);
  CHECK_INPUT(TK);
  CHECK_INPUT(TV);

  // cudaGuard用来确保在函数执行期间使用正确的CUDA设备
  at::cuda::CUDAGuard device_guard{TQ.device()};

  // 检查计算能力，确保当前设备支持所需的功能
  const int compute_capability =
      cuda_device_compute_capability(TQ.device().index());
  TORCH_CHECK(compute_capability >= 80,
              "Flash Attention requires SM_80 or higher (current: SM_",
              compute_capability / 10, ".", compute_capability % 10, ")");

  // 参数检查
  const auto Q_dtype = TQ.dtype();
  TORCH_CHECK(Q_dtype == torch::kFloat16 || Q_dtype == torch::kBFloat16,
              "Only fp16 and bf16 are supported");
  TORCH_CHECK(TK.dtype() == Q_dtype,
              "Input tensors must have the same data type");
  TORCH_CHECK(TV.dtype() == Q_dtype,
              "Input tensors must have the same data type");

  // 从Python配置对象转换为C++内核配置结构体
  const auto d_head = TQ.size(3);
  const FlashForwardKernelConfig cfg{py_to_cpp_kernel_config(py_cfg)};
  TORCH_CHECK(forward_kernels.contains(cfg),
              "Kernel configuration was not found in flash_kernels.cuh");

  // 根据配置选择对应的CUDA内核函数指针
  const auto kernel = forward_kernels[cfg];
  TORCH_CHECK(cfg.dtype == Q_dtype,
              "Kernel configuration dtype does not match input dtype");

  // 获取输入张量的维度信息
  const auto batch_size = TQ.size(0);
  const auto seq_len = TQ.size(1);
  const auto n_heads = TQ.size(2);

  // Only supported configuration currently.
  TORCH_CHECK(TQ.sizes() == TK.sizes(),
              "Query and key tensors have same shape");
  TORCH_CHECK(TQ.sizes() == TV.sizes(),
              "Query and value tensors have same shape");

  const int B_r = cfg.B_r;
  const int B_c = cfg.B_c;
  TORCH_CHECK(seq_len % B_r == 0,
              "Only multiples of B_r are supported for seq_len Q currently");
  TORCH_CHECK(seq_len % B_c == 0,
              "Only multiples of B_c are supported for seq_len K currently");

  // 计算输入张量的步幅信息，用于内核中的地址计算
  const auto batch_stride = TQ.stride(0);
  const auto seq_stride = TQ.stride(1);
  const auto head_stride = TQ.stride(2);

  // 准备输出张量，如果用户提供了输出张量则使用它，否则创建一个新的输出张量
  torch::Tensor TO;
  if (out_.has_value()) {
    TO = out_.value();
    TORCH_CHECK(TO.dtype() == Q_dtype,
                "Output tensor must have the same dtype as inputs");

    TORCH_CHECK(TQ.sizes() == TV.sizes(),
                "Query and output tensors have same shape");
  } else {
    TO = torch::empty_like(TQ);
  }

  // 计算块的数量和线程数量，以及softmax的缩放因子
  const int n_Q_blocks = CEIL_DIV(seq_len, B_r);
  const int n_KV_blocks = CEIL_DIV(seq_len, B_c);
  const int n_threads = cfg.n_warps * WARP_SIZE;
  float softmax_scale = M_LOG2E / sqrtf(d_head);

  // 将内核参数打包成结构体，传递给CUDA内核
  ForwardKernelArgs args{TQ.data_ptr(), TK.data_ptr(), TV.data_ptr(),
                         TO.data_ptr(), batch_stride,  seq_stride,
                         head_stride,   seq_len,       n_heads,
                         n_Q_blocks,    n_KV_blocks};

  // 定义CUDA内核的网格和块维度
  dim3 blockDim(n_threads);
  dim3 gridDim{static_cast<uint>(n_Q_blocks), static_cast<uint>(n_heads),
               static_cast<uint>(batch_size)};

  float runtime;
  cudaEvent_t start, stop;

  const int smem_bytes = cfg.smem_bytes();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (benchmark) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
  }
  kernel<<<gridDim, blockDim, smem_bytes, stream>>>(args);
  if (benchmark) {
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runtime, start, stop);
  }

  return std::make_tuple(TO, runtime);
}

} // namespace rapid_flash_attention