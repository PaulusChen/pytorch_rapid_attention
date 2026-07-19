#pragma once

#include "common.h"
#include <torch/torch.h>

namespace rapid_flash_attention {

struct FlashForwardKernelConfig {
  const torch::ScalarType dtype;
  const int d_head;  // [64, 128]
  const int B_r;     // [64, 128]
  const int B_c;     // [32, 64, 128]
  const int n_warps; // [4, 8]. 8 only when B_r = 128

  const bool async_copy;
  // If true, load K and V block tiles into smem as soon as we can.
  const bool eager_load_blocks;
  const bool swizzled;

  const int Q_mma_load_K_fragments;
  const int K_mma_load_K_fragments;
  const int V_mma_load_K_fragments;

  // if true, call ldmatrix for the next iter before calling mma.
  const bool mma_double_buffer_loads;
  const bool optimized_softmax;

  int smem_bytes(int elem_size = 2) const {
    return (B_r + B_c * 2) * d_head * elem_size;
  }

  int num_ctas_per_sm(int max_smem_bytes) const {
    // The max # ctas will be 2 or less due to register limits.
    if ((n_warps == 8) || (max_smem_bytes < smem_bytes() * 2)) {
      return 1;
    }

    return 2;
  }

  bool operator<(const FlashForwardKernelConfig &other) const {
    if (dtype != other.dtype) {
      return dtype < other.dtype;
    }
    if (d_head != other.d_head) {
      return d_head < other.d_head;
    }
    if (B_r != other.B_r) {
      return B_r < other.B_r;
    }
    if (B_c != other.B_c) {
      return B_c < other.B_c;
    }
    if (n_warps != other.n_warps) {
      return n_warps < other.n_warps;
    }
    if (async_copy != other.async_copy) {
      return async_copy < other.async_copy;
    }
    if (eager_load_blocks != other.eager_load_blocks) {
      return eager_load_blocks < other.eager_load_blocks;
    }
    if (swizzled != other.swizzled) {
      return swizzled < other.swizzled;
    }
    if (Q_mma_load_K_fragments != other.Q_mma_load_K_fragments) {
      return Q_mma_load_K_fragments < other.Q_mma_load_K_fragments;
    }
    if (K_mma_load_K_fragments != other.K_mma_load_K_fragments) {
      return K_mma_load_K_fragments < other.K_mma_load_K_fragments;
    }
    if (V_mma_load_K_fragments != other.V_mma_load_K_fragments) {
      return V_mma_load_K_fragments < other.V_mma_load_K_fragments;
    }
    if (mma_double_buffer_loads != other.mma_double_buffer_loads) {
      return mma_double_buffer_loads < other.mma_double_buffer_loads;
    }
    if (optimized_softmax != other.optimized_softmax) {
      return optimized_softmax < other.optimized_softmax;
    }
    return false; // Equal configurations
  }
};

template <int n, int K, bool double_buffer>
constexpr void static_assert_valid_load_k_fragments() {
  static_assert(((n & (n - 1)) == 0) && n != 1,
                "load k is power of 2 and DNE 1");

  constexpr int max_frags = (double_buffer ? K / 2 : K) / 8;
  static_assert(n <= max_frags, "load k is too large for K");
}

template <FlashForwardKernelConfig cfg> constexpr bool valid_config() {
  static_assert_valid_load_k_fragments<cfg.Q_mma_load_K_fragments, cfg.d_head,
                                       cfg.mma_double_buffer_loads>();

  static_assert_valid_load_k_fragments<cfg.K_mma_load_K_fragments, cfg.d_head,
                                       cfg.mma_double_buffer_loads>();

  static_assert_valid_load_k_fragments<cfg.V_mma_load_K_fragments, cfg.d_head,
                                       cfg.mma_double_buffer_loads>();

  static_assert((cfg.Q_mma_load_K_fragments == cfg.K_mma_load_K_fragments) ||
                cfg.Q_mma_load_K_fragments == 0);

  return true;
}

template <FlashForwardKernelConfig CFG> struct ForwardKernelTileShapes {
  static_assert(valid_config<CFG>());

  // 这个线程块（CTA）负责加载和处理的 d_head 分块（tile）数量。
  static constexpr int d_head_fragments = CFG.d_head / COLS_PER_FRAGMENT;
  static constexpr int d_head_accum_regs =
      d_head_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;

  // 每个 warp 独立加载和计算的 Q/O 行数或分块数（tiles），
  // 这对应一个大小为 (B_r / n_warps, d_head) 的数据块（chunk）。
  static constexpr int QO_rows_per_warp = CFG.B_r / CFG.n_warps;
  static constexpr int QO_fragments_per_warp =
      QO_rows_per_warp / ROWS_PER_FRAGMENT;

  // 对于 K/V 数据块，每个 warp 会独立加载一个 (B_c, d_head) 的分块（chunk），
  // 但计算时会基于线程块（CTA）加载的整个数据块进行。

  // 每个 warp 操作的 K/V 分块数量，这对应多个 (B_c, d_head) 大小的数据块。
  static constexpr int KV_calc_fragments = CFG.B_c / ROWS_PER_FRAGMENT;
  static constexpr int KV_calc_accum_regs =
      KV_calc_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;

  // 每个 warp 加载到共享内存（smem）中的 K/V 分块（tile）数量，
  // 这对应一个大小为 (B_c / n_warps, d_head) 的数据块（chunk）。
  static constexpr int KV_ldst_fragments_per_warp =
      KV_calc_fragments / CFG.n_warps;
  static constexpr int KV_ldst_rows_per_warp =
      KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;

  // 在 mma 指令之间的矩阵乘法期间要加载的分块（tiles）数量。
  static constexpr int Q_mma_load_K_fragments =
      CFG.Q_mma_load_K_fragments == 0 ? d_head_fragments
                                      : CFG.Q_mma_load_K_fragments;
  static constexpr int Q_mma_load_stages =
      (CFG.Q_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;

  static constexpr int K_mma_load_K_fragments =
      CFG.K_mma_load_K_fragments == 0 ? d_head_fragments
                                      : CFG.K_mma_load_K_fragments;
  static constexpr int K_mma_load_stages =
      (CFG.K_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;

  static constexpr int V_mma_load_K_fragments =
      (CFG.V_mma_load_K_fragments == 0 ? KV_calc_fragments
                                       : CFG.V_mma_load_K_fragments);
  static constexpr int V_mma_load_stages =
      (CFG.V_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;
};

struct LDSTCommon {
  const bool swizzled;
  const bool async_copy;
};

struct TileLayout {
  const int row_fragments;
  const int col_fragments;
};

// 包含 LDST（加载/存储）相关参数的 constexpr 非类型模板参数，
// 用于控制线程块（CTA）在全局内存（GMEM）与共享内存（SMEM）之间的数据搬运，
// 以及从共享内存（SMEM）到寄存器文件（RF）的加载操作。
struct TensorLDSTConfig {
  // 这包含了每个 warp 在全局内存（GMEM）和共享内存（SMEM）之间
  // 加载/存储（load/store）的 (8, 8) 分块（tile）数量。
  const TileLayout GSM;

  // 包含每个 warp 将计算（compute on）的片段（fragments）数量。
  const TileLayout RF;
  const LDSTCommon Common;
  const bool transposed;
  const int block_size;
  const int smem_cols;

  // 这是线程块（thread-block）中一个 warp 独立加载/存储的行数。
  // 它等于 GSM.row_fragments * 8
  const int warp_ldst_rows;

  // 该 warp 是否将计算整个 block（数据块）。
  // 对于（Q & O & S）为 false，对于（K & V）为 true.
  const bool compute_over_entire_block;

  // 该 warp 是否将加载整个 block（数据块）到寄存器文件（RF）。
  const bool load_entire_block_info_rf;

  const int mma_load_stages;
};

struct ForwardKernelArgs {
  using index_t = int64_t;

  void *__restrict__ Q;
  void *__restrict__ K;
  void *__restrict__ V;
  void *__restrict__ O;

  // We assume all strides are the same across all inputs, and that
  // the tensors are all row major.
  const index_t batch_stride;
  const index_t seq_stride;
  const index_t head_stride;

  const index_t seq_len;
  const index_t n_heads;

  const int n_Q_blocks;
  const int n_KV_blocks;
};

typedef void (*forward_kernel_fn)(const ForwardKernelArgs);

} // namespace rapid_flash_attention