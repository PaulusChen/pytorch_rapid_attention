#pragma once

#include "flash_attention_config.h"
#include <map>
#include <tuple>
#include <type_traits>

namespace rapid_flash_attention {

template <typename value_t>
__device__ void
mma_m16n8k16_f32_accum(
    float &d1, float &d2, float &d3, float &d4,
    uint32_t const &a1, uint32_t const &a2, uint32_t const &a3, uint32_t const &a4,
    uint32_t const &b1, uint32_t const &b2,
    float const &c1, float const &c2, float const &c3, float const &c4
) {
    static_assert(std::is_same_v<value_t, half> || std::is_same_v<value_t, nv_bfloat16>, "value_t must be half or nv_bloat16");
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

template <typename T>
__device__ void ldmatrix_x4(T *load_from, uint32_t &a1, uint32_t &a2,
                            uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);

    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

template <typename T>
__device__ void ldmatrix_x4_transpose(T *load_from, uint32_t &a1, uint32_t &a2, uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
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
struct GM2SM_async {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T*smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};

template <typename T>
struct GM2SM {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T*smem) {
        reinterpret_cast<uint4 *>(smem)[0] = reinterpret_cast<uint4 *>(gmem)[0];
    }
};

template <typename T>
struct SM2GM {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};

// 将一个 (B_r, d_head) 或 (B_c, d_head) 的数据块从全局内存（GMEM）拷贝到
// 共享内存（SMEM），或反向拷贝。
// 每个 warp 独立加载一个 (seq_len_per_warp, d_head) 的数据块。
// 每次内层迭代加载一个 (4, 64) 的分块（tile），其中每行由 8 个连续线程组成的组加载。
// 在边缘情况（edge case）下，如果要加载一个 (128, 64) 的数据块且有 8 个 warp，每个 warp
template <typename op, TensorLDSTConfig CFG, typename value_t, typename index_t = int64_t>
FA_DEVICE_CONSTEXPR void copy_block_GSM(value_t **gmem, value_t **smem,
                                        index_t gmem_seq_stride, const int lane_id) {
    constexpr int n_row_iters = CFG.GSM.row_fragments * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;

    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr int col_fragments_per_row = CFG.smem_cols / COLS_PER_FRAGMENT;

    const int thread_row = lane_id / col_fragments_per_iter;
    const int thread_col_fragment = lane_id % col_fragments_per_iter;

    FA_UNROLL
    for (int r = 0; r < n_row_iters; ++r) {
        const int cur_row = r * GSM_LDST_ROWS_PER_ITER + thread_row;
        FA_UNROLL
        for (int c = 0; c < col_fragments_per_row; c += col_fragments_per_iter) {
            const int gmem_col_fragment = c + thread_col_fragment;
            const int smem_col_fragment = get_smem_col_fragment<col_fragments_per_row, CFG.Common.swizzled>(cur_row, gmem_col_fragment);

            op() (&gmem[cur_row * gmem_seq_stride + gmem_col_fragment * COLS_PER_FRAGMENT],
                  &smem[cur_row * CFG.smem_cols + smem_col_fragment * COLS_PER_FRAGMENT]);
        }
    }
}

// 将共享内存（smem）中的矩阵分块载入通用寄存器
// 单条 ldmatrix.x4 指令可加载一块 16×16 的矩阵片段，等价于载入 2×2 个基础矩阵分片
// 本分支为**非转置**加载版本：共享内存存储排布与寄存器内存排布完全对应，满足：
// 寄存器内存维度 (r_r, r_c) = 共享内存行维度/8 , 共享内存列维度/8
// 该加载逻辑用于拷贝 Query 矩阵与 Key 矩阵
template <TensorLDSTConfig CFG, typename value_t>
FA_DEVICE_CONSTEXPR void copy_warp_fragments_SM2RF(uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments], value_t *smem,
                                                   const int lane_id, const int col_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.row_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                                                                cur_row, thread_col_fragment + c + col_fragment_offset);
            ldmatrix_x4(&smem[cur_row * CFG.smem_cols + smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                        regs[r][c], regs[r + 1][c], regs[r][c + 1],
                        regs[r + 1][c + 1]);
        }
    }
}

// 把共享内存(smem)中的矩阵分片加载到寄存器中
// 每条 ldmatrix.x4 指令加载一个 16×16 的数据块，也就是 2×2 个基础分片
// 此为转置加载版本：共享内存里矩阵的排布等价于寄存器内存矩阵的转置
// 即寄存器内存维度 rmem(r_r, r_c) = (smem列数 / 8, smem行数 / 8)
// 该逻辑用于拷贝 V（Value）矩阵
template <TensorLDSTConfig CFG, typename value_t>
FA_DEVICE_CONSTEXPR void copy_warp_fragment_transposed_SM2RF(uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments], value_t **smem,
                                                              const int lane_id, const int row_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.col_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.row_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(cur_row, thread_col_fragment + c);

            ldmatrix_x4_transpose(&smem[cur_row * CFG.smem_cols + smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                                  regs[c][r],
                                  regs[c][r + 1],
                                  regs[c + 1][r],
                                  regs[c + 1][r + 1]);
        }
    }
}

// 将寄存器文件（RF / rmem）中的矩阵片段（fragments）拷贝到共享内存（SMEM）。
// 内层循环的每次迭代拷贝一个 (8, 8) 的分块（tile），即一个片段（fragment）。
// 这将被用于拷贝 O（输出矩阵）。
template <TensorLDSTConfig CFG, typename value_t>
FA_DEVICE_CONSTEXPR void copy_warp_fragment_RF2SM(uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
                                                   value_t *smem,
                                                   const int lane_id) {
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT;
    constexpr int col_fragments_per_iter = 1;
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;

    constexpr int elems_per_store = 2;
    const int thread_row = lane_id / 4;
    const int thread_inner_col = (lane_id % 4) * elems_per_store;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.row_fragments; ++r) {
        const int cur_row = thread_row + r * rows_per_iter;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_frament = get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(cur_row, c);
            reinterpret_cast<uint32_t *>(&smem[cur_row * CFG.smem_cols + (smem_col_frament * ELEMS_PER_VEC4_ACCESS + thread_inner_col)])[0] = regs[r][c];
        }
    }
}

template <typename value_t, int M_fragments, int N_fragments>
FA_DEVICE_CONSTEXPR void convert_to_16_bit_dtype(float (&&src_float)[M_fragments][N_fragments * 2],
                                                 uint32_t (&dest_uint)[M_fragments][N_fragments]) {
    using value2_t = std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;

    float2(&src)[M_fragments][N_fragments] = reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);

    value2_t(&dest)[M_fragments][N_fragments] = reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);

    FA_UNROLL
    for (int m = 0; m < M_fragments; ++m) {
        FA_UNROLL
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}

template <typename value_t, int M_fragments, int N_fragments>
FA_DEVICE_CONSTEXPR void convert_to_16_bit_dtype(float (&src_float)[M_fragments][N_fragments * 2],
                                                 uint32_t (&dest_uint)[M_fragments][N_fragments]) {
    using value2_t = std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;

    float2(&src)[M_fragments][N_fragments] = reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    value2_t(&dest)[M_fragments][N_fragments] = reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);
    FA_UNROLL
    for (int m = 0; m < M_fragments; ++m) {
        FA_UNROLL
        for (int n = 0; n < N_fragments; ++n) {
            if  constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}

template <typename value_t, int N>
struct RFVector {
    static constexpr int size = N;
    value_t regs[N];
    FA_DEVICE_CONSTEXPR value_t &operator[](int idx) { return regs[idx]; }
};

template <typename value_t, int n_copies, int row_fragments, int col_fragments>
struct RFMatrix {
    using storage_t = std::conditional_t<sizeof(value_t) == 4, float, uint32_t>;
    static constexpr int regs_per_fragment = sizeof(value_t) / 2;
    static constexpr int rows = row_fragments;
    static constexpr int cols = col_fragments * regs_per_fragment;

    storage_t regs[n_copies][rows][cols];

    FA_DEVICE_CONSTEXPR storage_t (&data(const int stage = 0))[rows][cols] {
        return reinterpret_cast<storage_t(&)[rows][cols]>(regs[stage]);
    }

    FA_DEVICE_CONSTEXPR void zero() {
        FA_UNROLL
        for (int i = 0; i < n_copies; ++i) {
            FA_UNROLL
            for (int j = 0; j < rows; ++j) {
                FA_UNROLL
                for (int k = 0; k < cols; ++k) {
                    regs[i][j][k] = 0;
                }
            }
        }
    }
};

// MatrixLDST 是一个对象，它为内存中的一个数据块（block）提供加载/存储（LDST）和类型转换（conversion）功能。
// 该对象的作用范围涵盖所有内存层级（全局内存 GMEM、共享内存 SMEM 和寄存器文件 RF）。
// 诚然，这个类承担了过多职责，但考虑到本项目的范围，我不想过度设计它。
template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    using matrix_storage_t = RFMatrix<value_t, ldst.mma_load_stages, ldst.RF.row_fragments, ldst.RF.col_fragments>;
    using GM2SM_op = std::conditional_t<ldst.Common.async_copy, GM2SM_async<value_t>, GM2SM<value_t>>;

    using SM2GM_op = SM2GM<value_t>;
    static constexpr int mma_load_stages = ldst.mma_load_stages;
    static constexpr bool load_entire_block_into_rf = ldst.load_entire_block_info_rf;
    static constexpr bool transposed = ldst.transposed;

    value_t *gmem_ptr;
    index_t gmem_seq_stride;

    //用于将数据分片从共享内存（smem）加载至寄存器内存（rmem）的内存寻址位置
    value_t *smem_srm_ptr;

    // 该内存地址用于线程束将Q、K、V从全局内存写入共享内存，
    // 同时也用于将输出矩阵O从共享内存(smem)写回全局内存(gmem)。
    value_t *smem_gsm_ptr;

    const int lane_id;

    matrix_storage_t storage;

    FA_DEVICE MatrixLDST(value_t *gmem_block_ptr,
                         index_t _gmem_seq_stride,
                         value_t * _smem_ptr) : lane_id(threadIdx.x % WARP_SIZE) {
        const int warp_rank = threadIdx.x / WARP_SIZE;
        const index_t warp_seq = ldst.warp_ldst_rows * warp_rank;

        gmem_seq_stride = _gmem_seq_stride;
        gmem_ptr = gmem_block_ptr + warp_seq * gmem_seq_stride;

        smem_gsm_ptr = _smem_ptr + warp_seq * ldst.smem_cols;
        smem_srm_ptr = ldst.compute_over_entire_block ? _smem_ptr : smem_gsm_ptr;
    }

    FA_DEVICE_CONSTEXPR void zero() { storage.zero(); }

    FA_DEVICE_CONSTEXPR typename matrix_storage_t::storage_t (&data(const int stage = 0))[matrix_storage_t::rows][matrix_storage_t::cols] {
        return storage.data(stage);
    }

    FA_DEVICE_CONSTEXPR void advance_gmem_block() {
        gmem_ptr += ldst.block_size * gmem_seq_stride;
    }

    FA_DEVICE_CONSTEXPR void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride, lane_id);
    }

    FA_DEVICE_CONSTEXPR void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride, lane_id);
    }

    FA_DEVICE_CONSTEXPR void copy_SM2RF(int stage = 0, int tile_offset = 0) {
        if constexpr (!transposed) {
            copy_warp_fragment_SM2RF<ldst, value_t>(storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        } else {
            // 转置
            copy_warp_fragment_transposed_SM2RF<ldst, value_t>(storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        }
    }

    FA_DEVICE_CONSTEXPR void copy_RF2SM() {
        copy_warp_fragment_RF2SM<ldst, value_t>(data(), smem_srm_ptr, lane_id);
    }
};

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define MMA_M_FRAGMENTS_PER_ITER 2 // (MMA_M / LDMATRIX_MAT_SIZE)
#define MMA_N_FRAGMENTS_PER_ITER 1 // (MMA_N / LDMATRIX_MAT_SIZE)
#define MMA_K_FRAGMENTS_PER_ITER 2 // (MMA_K / LDMATRIX_MAT_SIZE)

template<typename _A_t, typename _B_t, typename _C_t, int total_K_fragments, int load_K_fragments_per_iter, typename value_t_>
struct GEMM {
    using A_t = _A_t;
    using B_t = _B_t;
    using C_t = _C_t;
    using value_t = value_t_;

    static constexpr int TotalKTiles = total_K_fragments;
    static constexpr int LoadKtilesPerIter = load_K_fragments_per_iter;

    static constexpr bool DoubleBufferA = !A_t::load_entire_block_into_rf && A_t::mma_load_stages > 1;
    static constexpr bool DoubleBufferB = !B_t::load_entire_block_into_rf && B_t::mma_load_stages > 1;
    static constexpr bool DoubleBuffer = DoubleBufferA || DoubleBufferB;
};

template<typename value_t, const int M_fragments, const int N_fragments, const int K_fragments_A, const int K_fragments_B, typename accum_t = float>
FA_DEVICE_CONSTEXPR void warp_fragment_mma_f32_accum(uint32_t (&regs_A)[M_fragments][K_fragments_A],
                                                     uint32_t (&regs_B)[N_fragments][K_fragments_B],
                                                     accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT],
                                                     int A_col_fragment_offset = 0, int B_col_fragment_offset = 0) {
    constexpr int K_iters = constexpr_min(K_fragments_A, K_fragments_B);
    FA_UNROLL
    for (int k = 0; k < K_iters; k += MMA_K_FRAGMENTS_PER_ITER) {
        FA_UNROLL
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            FA_UNROLL
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k + A_col_fragment_offset],
                    regs_A[m + 1][k + A_col_fragment_offset],
                    regs_A[m][k + 1 + A_col_fragment_offset],
                    regs_A[m + 1][k + 1 + A_col_fragment_offset],
                    regs_B[n][k + B_col_fragment_offset],
                    regs_B[n][k + 1 + B_col_fragment_offset],
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

template <typename GEMM>
FA_DEVICE_CONSTEXPR void matmul(typename GEMM::A_t &A,
                                typename GEMM::B_t &B,
                                typename GEMM::C_t &C) {
    using A_t = typename GEMM::A_t;
    using B_t = typename GEMM::B_t;
    using value_t = typename GEMM::value_t;

    constexpr int A_stage_toggle = A_t::mma_load_stages - 1;
    constexpr int B_stage_toggle = B_t::mma_load_stages - 1;

    int A_stage = 0;
    int B_stage = 0;

    if constexpr (GEMM::DoubleBufferA) {
        A.copy_SM2RF(A_stage);
    }

    if constexpr (GEMM::DoubleBufferB) {
        B.copy_SM2RF(B_stage);
    }

    FA_UNROLL
    for (int k_outer_fragment = 0; k_outer_fragment < GEMM::TotalKTiles; k_outer_fragment += GEMM::LoadKTilesPerIter) {
        if constexpr (!A_t::load_entire_block_into_rf || !B_t::load_entire_block_into_rf) {
            int k_load_fragment = k_outer_fragment + (GEMM::DoubleBuffer ? GEMM::LoadKTilesPerIter : 0);
            if (k_load_fragment < GEMM::TotalKTiles) {
                if constexpr (!A_t::load_entire_block_info_rf) {
                    A.copy_SM2RF(A_stage_toggle ^ A_stage, k_load_fragment);
                }
                if constexpr (!B_t::load_entire_block_info_rf) {
                    B.copy_SM2RF(B_stage_toggle ^ B_stage, k_load_fragment);
                }
            }
        }

        int A_col_offset = A_t::load_entire_block_into_rf ? k_outer_fragment : 0;
        int B_col_offset = B_t::load_entire_block_into_rf ? k_outer_fragment : 0;
        warp_fragment_mma_f32_accm<value_t>(A.data(A_stage), B.data(B_stage), C.data(), A_col_offset, B_col_offset);
        A_stage ^= A_stage_toggle;
        B_stage ^= B_stage_toggle;
    }
}

template <FlashForwardKernelConfig CFG> struct StaticForwardKernelConfig {
  using accum_t = float;
  using value_t = typename ::std::conditional_t<CFG.dtype == torch::kBFloat16,
                                                nv_bfloat16, half>;

  using N = ForwardKernelTileShapes<CFG>;

  static constexpr bool async_copy = CFG.async_copy;
  static constexpr int B_r = CFG.B_r;
  static constexpr int B_c = CFG.B_c;
  static constexpr int d_head = CFG.d_head;
  static constexpr bool eager_load_blocks = CFG.eager_load_blocks;
  static constexpr bool optimized_softmax = CFG.optimized_softmax;

  static constexpr LDSTCommon Common{CFG.swizzled, CFG.async_copy};

  static constexpr TensorLDSTConfig make_ldst_config(
      TileLayout GSM, TileLayout RF, bool transposed, int block_size,
      int warp_ldst_rows, bool compute_over_entire_block,
      bool load_entire_block_into_rf = true, int mma_load_stages = 1) {
    return TensorLDSTConfig{GSM,
                            RF,
                            Common,
                            transposed,
                            block_size,
                            CFG.d_head,
                            warp_ldst_rows,
                            compute_over_entire_block,
                            load_entire_block_into_rf,
                            mma_load_stages};
  }

  static constexpr TensorLDSTConfig Q_LDST = make_ldst_config(
                                                    { N::QO_fragments_per_warp, N::d_head_fragments },
                                                    { N::QO_fragments_per_warp, N::Q_mma_load_K_fragments },
                                                    false,
                                                    CFG.B_r,
                                                    N::QO_rows_per_warp,
                                                    false,
                                                    CFG.Q_mma_load_K_fragments == 0,
                                                    N::Q_mma_load_stages);
  using Q_t = MatrixLDST<Q_LDST, value_t>;

  static constexpr TensorLDSTConfig K_LDST = make_ldst_config(
                                                    { N::KV_ldst_fragments_per_warp, N::d_head_fragments},
                                                    { N::KV_calc_fragments, N::K_mma_load_K_fragments},
                                                    false,
                                                    CFG.B_c,
                                                    N::KV_ldst_rows_per_warp,
                                                    true,
                                                    CFG.K_mma_load_K_fragments == 0,
                                                    N::K_mma_load_stages);
  using K_t = MatrixLDST<K_LDST, value_t>;

  static constexpr TensorLDSTConfig V_LDST = make_ldst_config(
                                                    { N::KV_ldst_fragments_per_warp, N::d_head_fragments},
                                                    { N::d_head_fragments, N::V_mma_load_K_fragments },
                                                    true,
                                                    CFG.B_c,
                                                    N::KV_ldst_rows_per_warp,
                                                    true,
                                                    CFG.V_mma_load_K_fragments == 0,
                                                    N::V_mma_load_stages);
  using V_t = MatrixLDST<V_LDST, value_t>;

  static constexpr TensorLDSTConfig O_LDST = make_ldst_config(
                                                    { N::QO_fragments_per_warp, N::d_head_fragments },
                                                    { N::QO_fragments_per_warp, N::d_head_fragments },
                                                    false,
                                                    CFG.B_r,
                                                    N::QO_rows_per_warp,
                                                    false,
                                                    true);
  using O_accum_t = MatrixLDST<O_LDST, accum_t>;
  using O_value_t = MatrixLDST<O_LDST, value_t>;

  // S/P 在内核的整个执行期间完全保持在寄存器文件（RF）中。
  static constexpr TensorLDSTConfig S_LDST = make_ldst_config(
                                                    { N::QO_fragments_per_warp, N::KV_calc_fragments }, 
                                                    { N::QO_fragments_per_warp, N::KV_calc_fragments }, 
                                                    CFG.B_r,
                                                    false,
                                                    0,
                                                    false);
  using S_accum_t = MatrixLDST<S_LDST, accum_t>;
  using P_value_t = MatrixLDST<S_LDST, value_t>;

  using S_QK_GEMM = GEMM<Q_t, K_t, S_accum_t, N::d_head_fragments, constexpr_min(N::Q_mma_load_K_fragments, N::K_mma_load_K_fragments), value_t>;
};

} // rapid_flash_attention

