#pragma once

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
}

