#pragma once

#include <cuda_runtime.h>

#include <climits>
#include <type_traits>

namespace algo::cuda::scan {

template <class T>
struct Plus {
    __host__ __device__ constexpr T operator()(T lhs, T rhs) const {
        return lhs + rhs;
    }

    __host__ __device__ static constexpr T identity() {
        return T{0};
    }
};

template <class T>
struct Max {
    static_assert(std::is_integral_v<T>, "Max currently supports integral types");

    __host__ __device__ constexpr T operator()(T lhs, T rhs) const {
        return lhs < rhs ? rhs : lhs;
    }

    __host__ __device__ static constexpr T identity() {
        if constexpr (std::is_unsigned_v<T>) {
            return T{0};
        } else {
            using U = std::make_unsigned_t<T>;
            return static_cast<T>(U{1} << (sizeof(T) * CHAR_BIT - 1));
        }
    }
};

} // namespace algo::cuda::scan
