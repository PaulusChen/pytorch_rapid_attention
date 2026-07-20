#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <map>

#include "common.h"
#include "flash_attention_config.h"
#include "flash_attention_kernel.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace rapid_flash_attention {

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_S_accum(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
              const accum_t &softmax_scale) {
  FA_UNROLL
  for (int q = 0; q < QO_fragments; ++q) {
    FA_UNROLL
    for (int k = 0; k < KV_accum_fragments; ++k) {
      S_accum[q][k] *= softmax_scale;
    }
  }
}

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
calc_row_max(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
             accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments]) {
  FA_UNROLL
  for (int q = 0; q < QO_fragments; ++q) {
    m_next[q] = m_cur[q];

    FA_UNROLL
    for (int k = 0; k < KV_accum_fragments; ++k) {
      m_next[q] = max(m_next[q], S_accum[q][k]);
    }

    m_next[q] =
        max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2), m_next[q]);
    m_next[q] =
        max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1), m_next[q]);
  }
}

template <bool optimized_softmax, int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_l_O(accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments],
          accum_t (&l)[QO_fragments],
          accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
          accum_t softmax_scale) {
  FA_UNROLL
  for (int q = 0; q < QO_fragments; ++q) {
    accum_t scale;
    if constexpr (optimized_softmax) {
      scale = exp2f((m_cur[q] - m_next[q]) * softmax_scale);
    } else {
      scale = expf(m_cur[q] - m_next[q]);
    }
    m_cur[q] = m_next[q];
    l[q] *= scale;
    for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
      O_accum[q][d_head] *= scale;
    }
  }
}

template <bool optimized_softmax, int QO_fragments, int KV_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
exponentiate_tensor(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
                    accum_t (&m)[QO_fragments], accum_t softmax_scale) {
  FA_UNROLL
  for (int q = 0; q < QO_fragments; ++q) {
    accum_t max_scaled;
    if constexpr (optimized_softmax) {
      max_scaled = m[q] * softmax_scale;
    }
    FA_UNROLL
    for (int k = 0; k < KV_accum_fragments; ++k) {
      if constexpr (optimized_softmax) {
        S_accum[q][k] = exp2f(S_accum[q][k] * softmax_scale - max_scaled);
      } else {
        S_accum[q][k] = expf(S_accum[q][k] - m[q]);
      }
    }
  }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
update_row_exp_sum(accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
                   accum_t (&l)[QO_fragments]) {
  FA_UNROLL
  for (int q = 0; q < QO_fragments; ++q) {
    FA_UNROLL
    for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
      l[q] += P_accum[q][d_head];
    }
  }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void final_softmax_normalization(
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
  FA_UNROLL
  for (int q = 0; q < QO_fragments; ++q) {
    l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
    l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
  }

  FA_UNROLL
  for (int q = 0; q < QO_fragments; ++q) {
    FA_UNROLL
    for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
      O_accum[q][d_head] /= l[q];
    }
  }
}

template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {
  using accum_t = float;
  using index_t = int64_t;
  using N = typename Kernel::N;

  using value_t = typename Kernel::value_t;
  using Q_t = typename Kernel::Q_t;
  using K_t = typename Kernel::K_t;
  using V_t = typename Kernel::V_t;

  constexpr int async = Kernel::async_copy;

  // 我们为每个样本（sample）、序列分块（seq tile）和注意力头（head）初始化一个
  // CTA。
  const int sample = blockIdx.z;
  const int head = blockIdx.y;
  const int q_seq_block = blockIdx.x;

  const index_t gmem_seq_stride = args.seq_stride;

  const index_t sample_head_offset =
      sample * args.batch_stride + head * args.head_stride;

  // Q矩阵与输出矩阵O，本线程块仅读写单个分块数据。
  // 下述偏移量对整个线程块内所有线程统一生效。
  const index_t QO_gmem_block_offset =
      sample_head_offset + q_seq_block * Kernel::B_r * gmem_seq_stride;

  // 本线程块会读取Key的完整序列片段
  const index_t KV_gmem_block_offset = sample_head_offset;

  value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_gmem_block_offset];
  value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_gmem_block_offset];
  value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_gmem_block_offset];
  value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_gmem_block_offset];

  extern __shared__ __align__(16) char ch_smem[];
  value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
  value_t *smem_O = smem_Q;
  value_t *smem_K = smem_Q;
  value_t *smem_V = smem_K;

  Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);
  K_t K(gmem_K, gmem_seq_stride, smem_K);
  V_t V(gmem_V, gmem_seq_stride, smem_V);

  typename Kernel::S_accum_t S_accum(nullptr, -1, nullptr);
  typename Kernel::P_value_t P_b16(nullptr, -1, nullptr);
  typename Kernel::O_accum_t O_accum(nullptr, -1, nullptr);
  typename Kernel::O_value_t O_b16(gmem_O, gmem_seq_stride, smem_O);

  Q.copy_GM2SM();
  cp_async_commit<async>();
  if constexpr (Kernel::eager_load_blocks) {
    K.copy_GM2SM();
    K.advance_gmem_block();
    cp_async_commit<async>();
  }

  O_accum.zero();

  const accum_t softmax_scale = rsqrt(static_cast<accum_t>(Kernel::d_head)) *
                                (Kernel::optimized_softmax ? M_LOG2E : 1.0);
  constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();
  accum_t m[N::QO_fragments_per_warp];
  accum_t l[N::QO_fragments_per_warp];

  FA_UNROLL
  for (int q = 0; q < N::QO_fragments_per_warp; ++q) {
    m[q] = neg_inf;
    l[q] = 0.0;
  }

  if constexpr (Q_t::load_entire_block_into_rf) {
    if constexpr (Kernel::eager_load_blocks) {
      // 我们只等待 Q 数据块完成加载。
      cp_async_wait<1, async>();
    } else {
      cp_async_wait<0, async>();
    }

    // 除了 cp_async_wait() 之外，我们还需要 __syncwarp()，
    // 因为 cp_async_wait() 只阻塞当前线程，直到其数据加载完成。
    // 由于整个 warp 都会从共享内存（SMEM）读取这个数据块，
    // 我们需要等待一个 warp 范围内的屏障（barrier）。
    // 对于 K 和 V，我们将需要使用 __syncthreads() 替代。
    __syncwarp();
    Q.copy_SM2RF();
  }

  for (int j = 0; j < args.n_KV_blocks; ++j) {
    if constexpr (!Kernel::eager_load_blocks) {
      K.copy_GM2SM();
      K.advance_gmem_block();
      cp_async_commit<async>();
    }
    // 将 S 的寄存器初始化为 0。
    S_accum.zero();

    // 阻塞直到本轮迭代所需的 K
    // 数据块分块（block-tile）已被拷贝到共享内存（SMEM）中。
    cp_async_wait<0, async>();

    // 在此屏障之后，可以安全地加载下一个 V 数据块（V block），
    // 因为所有 warp 都已经完成了上一轮的 P × V 矩阵乘法（PV matmul）。
    __syncthreads();

    if constexpr (Kernel::eager_load_blocks) {
      // 启动 V 矩阵从 GMEM 到 SMEM 的（异步）拷贝，
      // 但不等它完成，而是等到 S = Q·K^T 矩阵乘法之后。
      if (j < args.n_KV_blocks - 1) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit<async>();
      }
    }

    accum_t m_next[N::QO_fragments_per_warp];
    if constexpr (!Kernel::optimized_softmax) {
      scale_S_accum(S_accum.data(), softmax_scale);
    }

    calc_row_max(S_accum.data(), m_next, m);
    scale_l_O<Kernel::optimized_softmax>(m_next, m, l, O_accum.data(),
                                         softmax_scale);
    exponentiate_tensor<Kernel::optimized_softmax>(S_accum.data(), m_next,
                                                   softmax_scale);
    update_row_exp_sum(S_accum.data(), l);

    convert_to_16_bit_dtype<value_t>(S_accum.data(), P_b16.data());

    if constexpr (!Kernel::eager_load_blocks) {
      // 将 V 从全局内存（GMEM）加载到共享内存（SMEM），并阻塞直到加载完成
      V.copy_GM2SM();
      V.advance_gmem_block();
      cp_async_commit<async>();
      cp_async_wait<0, async>();
      __syncthreads();
    }
    if constexpr (V_t::load_entire_block_into_rf) {
      V.copy_SM2RF();
    }
    matmul<typename Kernel::O_PV_GEMM>(P_b16, V, O_accum);
  }

  final_softmax_normalization(O_accum.data(), l);

  convert_to_16_bit_dtype<value_t>(O_accum.data(), O_b16.data());

  // 我们不直接写入全局内存（gmem），而是将共享内存（smem）作为中间步骤进行写入。
  // 这样我们可以：
  // - 使用 16 字节向量化存储，而非 4 字节存储
  // - 实现存储的完全合并访问（full coalescing）
  //   - 每个 warp 可以存储 4 条 128 字节对齐的缓存行（512 字节/warp），
  //   而不是 8 条 16 字节未合并的行（128 字节/warp）
  O_b16.copy_RF2SM();

  // 等待同一 warp 中的所有线程都已完成向共享内存（SMEM）的写入。
  // 此处不需要 __syncthreads()，因为各 warp 操作的是 O
  // 矩阵中相互独立的数据分块（chunks）。
  __syncwarp();

  // 将最终的 O 分块（O tile）从共享内存（SMEM）拷贝到全局内存（GMEM）
  O_b16.copy_SM2GM();
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

} // namespace rapid_flash_attention