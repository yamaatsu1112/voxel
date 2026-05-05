#pragma once

namespace algo::cuda::scan::detail {

inline constexpr int kWarpSize = 32;
inline constexpr unsigned int kWarpFullMask = 0xffffffffu;

template <class T>
using warp_shuffle_carrier_t = std::conditional_t<
    std::is_floating_point_v<T>,
    std::conditional_t<(sizeof(T) <= sizeof(float)), float, double>,
    std::conditional_t<(sizeof(T) <= sizeof(std::uint32_t)),
                       std::conditional_t<std::is_signed_v<T>, int,
                                          unsigned int>,
                       std::conditional_t<std::is_signed_v<T>, long long,
                                          unsigned long long>>>;

template <class T>
__device__ __forceinline__ warp_shuffle_carrier_t<T> to_warp_carrier(T value) {
    return static_cast<warp_shuffle_carrier_t<T>>(value);
}

template <class T>
__device__ __forceinline__ T from_warp_carrier(
    warp_shuffle_carrier_t<T> value) {
    return static_cast<T>(value);
}

template <class Op, class T>
__device__ __forceinline__ T warp_inclusive_scan(T value) {
    T sum_value = value;
    const Op op{};
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const auto other =
            __shfl_up_sync(kWarpFullMask, to_warp_carrier(sum_value), offset);
        if ((threadIdx.x & (kWarpSize - 1)) >= offset) {
            sum_value = op(from_warp_carrier<T>(other), sum_value);
        }
    }
    return sum_value;
}

} // namespace algo::cuda::scan::detail
