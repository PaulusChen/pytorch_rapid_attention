#pragma once

namespace rapid_flash_attention {

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                             \
    {                                                                          \
        auto status = static_cast<cudaError_t>(error);                         \
        if (status != cudaSuccess) {                                           \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":"  \
                      << __LINE__ << std::endl;                                \
            std::exit(status);                                                 \
        }                                                                      \
    }
#endif

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__device__ __forceinline__ bool is_cta_leader() { return threadIdx.x == 0; }

inline int cuda_device_num_sms(int device) {
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    return sms;
}

inline int cuda_device_max_smem_bytes(int device) {
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    return max_smem;
}

inline int cuda_device_compute_capability(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major * 10 + prop.minor;
}

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