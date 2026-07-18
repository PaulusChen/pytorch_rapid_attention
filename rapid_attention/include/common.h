
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#ifdef FA_DEBUG
#define FA_UNROLL
#else
#define FA_UNROLL _Pragma("unroll")
#endif

#define FA_DEVICE __forceinline__ __device__
#define FA_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr

#define N_HEADS 16

#define WARP_SIZE 32
#define SHFL_ENTIRE_WARP_MASK 0xffffffff

#define B16_BYTES 2
#define BYTES_PER_VEC4_ACCESS 16
#define ELEMS_PER_VEC4_ACCESS (BYTES_PER_VEC4_ACCESS / B16_BYTES)

// mma/ldmatrix related constants
#define MMA_A_REGS_PER_ROW 2
#define MMA_A_REGS_PER_COL 2
#define MMA_B_REGS_PER_ROW 2
#define MMA_B_REGS_PER_COL 1
#define MMA_C_REGS_PER_ROW 1
#define MMA_C_REGS_PER_COL 2

#define N_REGS_PER_F32_ACCUM_FRAGMENT 2

#define LDMATRIX_MAT_SIZE 8
#define ROWS_PER_FRAGMENT LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT LDMATRIX_MAT_SIZE

#define GSM_LDST_ROWS_PER_ITER 4

#define N_BUFFER_STAGES 2

#define GSMEM_THR_PER_ROW 8
#define SWIZZLE_TILE_SIZE 64

namespace rapid_flash_attention {

struct alignas(16) uint128_t {
    uint64_t low;
    uint64_t high;
};

template <typename value_t>
constexpr bool is_supported_mma_input_type() {
    return std::is_same_v<value_t, half> ||
           std::is_same_v<value_t, nv_bfloat16>;
}

template <typename value_t>
constexpr bool is_supported_mma_output_type() {
    return std::is_same_v<value_t, float>;
}

template <typename value_t>
constexpr auto value_storage_type() {
    if constexpr (is_supported_mma_input_type<value_t>()) {
        return uint32_t{};
    } else if constexpr (is_supported_mma_output_type<value_t>()) {
        return float{};
    }
}

template <typename value_t>
constexpr auto value2_storage_type() {
    if constexpr (std::is_same_v<value_t, half>) {
        return half2{};
    } else if constexpr (std::is_same_v<value_t, nv_bfloat16>) {
        return nv_bfloat162{};
    } else if constexpr (std::is_same_v<value_t, float>) {
        return float2{};
    }
}

constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }

constexpr int constexpr_max(int a, int b) { return (a > b) ? a : b; }

constexpr int constexpr_log2_floor(int n) { return std::__bit_width(n) - 1; }

constexpr int binary_to_pm1(int b) { return 2 * b - 1; }

template <int N, typename value_t_>
struct Array {
    using value_t = value_t_;

    value_t _data[N];

    FA_DEVICE_CONSTEXPR value_t *data() { return _data; }
    FA_DEVICE_CONSTEXPR const value_t *data() const { return _data; }

    FA_DEVICE_CONSTEXPR void fill(value_t val) {
        FA_UNROLL
        for (int i = 0; i < N; ++i) {
            _data[i] = value_t(val);
        }
    }

    FA_DEVICE_CONSTEXPR void zero() { fill(0); }

    FA_DEVICE_CONSTEXPR value_t &operator[](size_t idx) { return _data[idx]; }
    FA_DEVICE_CONSTEXPR value_t operator[](size_t idx) const {
        return _data[idx];
    }

    FA_DEVICE_CONSTEXPR static size_t size() { return N; }

    template <typename Other>
    FA_DEVICE_CONSTEXPR void copy(const Other &other) {
        static_assert(std::is_same<value_t, typename Other::value_t>::value,
                      "Arrays must have the same value type");
        static_assert(N == Other::size(), "Arrays must have the same size");

        FA_UNROLL
        for (int i = 0; i < N; ++i) {
            _data[i] = other[i];
        }
    }
};

template <int N, typename value_t, int Alignment = 16>
struct __align__(Alignment) ArrayAligned : public Array<N, value_t> {};

} // namespace rapid_flash_attention
