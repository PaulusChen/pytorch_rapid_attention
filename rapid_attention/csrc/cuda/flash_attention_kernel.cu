#include <map>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "flash_attention_config.h"
#include "flash_attention_kernel.cuh"

namespace rapid_flash_attention {

template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {
    using accum_t = float;
    using index_t = int64_t;
    using N = typename Kernel::N;

    using value_t = typename Kernel::value_t;
    using Q_t = typename Kernel::Q_t;

    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    const index_t gmem_seq_stride = args.seq_stride;
    const index_t sample_head_offset = sample * args.batch_stride + head * args.head_stride;

    const index_t QO_gmem_block_offset = sample_head_offset + q_seq_block * Kernel::B_r * gmem_seq_stride;
    
    const index_t KV_gmem_block_offset = sample_head_offset;

    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = smem_Q;
    value_t *smem_V = smem_K;
    // Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);

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


}