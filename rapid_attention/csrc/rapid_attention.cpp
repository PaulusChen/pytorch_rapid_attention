
#include <torch/extension.h> 
#include <cuda_runtime.h>

#include "cuda/flash_attention_operator.hpp"
#include "cuda/flash_attention_kernel.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 将 C++ 的 Flash Attention 前向函数绑定到 Python 模块中
  m.def("forward", &rapid_flash_attention::flash_attention_forward,
        py::arg("kernel_cfg"), py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("o"), py::arg("benchmark") = false,
        "Flash Attention forward (CUDA)");
  // 为每个内核配置设置动态共享内存大小属性
  for (const auto &[cfg, kernel] : rapid_flash_attention::forward_kernels) {
    int smem_used = cfg.smem_bytes();
    if (smem_used > 48 * 1024) {
      cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_used);
    }
  }
}
