#include "flash_attention_operator.hpp"

#include "common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <type_traits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace rapid_flash_attention {

template <typename value_t>
__device__ void
mma_m16n8k16_f32_accum(
    float &d1, float &d2, float &d3, float &d4,
    uint32_t const &a1, uint32_t const &a2, uint32_t const &a3, uint32_t const &a4,
    uint32_t const &b1, uint32_t const &b2,
    float const &c1, float const &c2, float const &c3, float const &c4
) {
    static_assert(std::is_same_v<value_t, half> || std::is_same_v<value_t, nv_bloat16>, "value_t must be half or nv_bloat16");
    if constexpr (std::is_same_v<value_t, nv_bfloat16>) {
        /*
        mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 d, a, b, c; 
        mma.sync.aligned.m16n8k8.row.col.f32.atype.btype.f32 d, a, b, c; 
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 d, a, b, c; 
        mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype d, a, b, c; 
        mma.sync.aligned.m16n8k32.row.col.kind.dtype.f8f6f4type.f8f6f4type.ctype d, a, b, c; 
        .atype = {.bf16, .tf32};
        .btype = {.bf16, .tf32}; 
        .f8type = {.e4m3, .e5m2}; 
        .f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1}; 
        .ctype = {.f16, .f32}; 
        .dtype = {.f16, .f32}; 
        .shape = {.m16n8k16, 
        .m16n8k32}; 
        .kind = {.kind::f8f6f4};
        */
        asm volatile("mma.sync,aligned.m16n8k16.row.col.f32.b16.b16.f32 "
                    " {%0, %1, %2, %3}, "
                    " {%4, %5, %6, %7}, "
                    " {%8, %9}, "
                    " {%10, %11, %12, %13}; "
                    : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                    : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
                    "r"(b1), "r"(b2),
                    "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        /*
        Half precision floating point type: 
        mma.sync.aligned.m8n8k4.alayout.blayout.dtype.f16.f16.ctype d, a, b, c; 
        mma.sync.aligned.m16n8k8.row.col.dtype.f16.f16.ctype d, a, b, c; 
        mma.sync.aligned.m16n8k16.row.col.dtype.f16.f16.ctype d, a, b, c; 
        .alayout = {.row, .col}; 
        .blayout = {.row, .col}; 
        .ctype = {.f16, .f32}; 
        .dtype = {.f16, .f32};
        */
        asm volatile("mma.sync,aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    " {%0, %1, %2, %3}, "
                    " {%4, %5, %6, %7}, "
                    " {%8, %9}, "
                    " {%10, %11, %12, %13}; "
                    : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                    : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
                    "r"(b1), "r"(b2),
                    "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

__device__ void cp_async_commit() {
    // 划清批次界限。它将自上一个 commit_group 以来发起的所有 cp.async 操作，打包成一个新的“group”
    asm volatile("cp.async.commit_group;");
}

template <int ngroups>
__device__ void cp_async_wait() {
    // cp.async.wait_group 指令将使执行线程等待，
    // 直到最近 N 个或更少的 cp.async 组（cp-async-groups）处于未完成（pending）状态，
    // 并且所有由执行线程提交的更早的 cp.async 组都已完成。
    // N 的本质，是定义了一个“滑动窗口”，决定了有多少批次的异步拷贝任务可以“在途”（尚未完成），从而实现计算和拷贝的重叠。
    asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
}

template <int size, typename T>
__device__ void cp_async(T *smem_to, const T *gmem_from) {
    static_assert(size == 16);

    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_from), "n"(size));
}

template <typename T>
__device__ void ldmatrix_x4(T *load_from, uint32_t &a1, uint32_t &a2,
                            uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);

    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
    

}

template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {


}

std::map<FlashForwardKernelConfig, forward_kernel_fn> forward_kernels = {

    {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true,
                              true, 2, 2, 2, true, true},
     &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{
         torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true,
         true}>>},
    {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                              2, 2, 2, true, true},
     &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{
         torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true,
         true}>>}};

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