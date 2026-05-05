#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix.cuh>

namespace algo::cuda::sort::detail {

template <class Config> struct workspace_layout;

template <class Config, class Value> struct pair_workspace_layout;

template <class ScanPolicy> struct flag_scan_impl;

template <class ScanPolicy> struct scatter_impl;

template <class HistogramPolicy> struct histogram_build_impl;

template <class WarpRankPolicy> struct warp_rank_impl;

template <class LocalRankPolicy> struct histogram_local_rank_impl;

template <class LocalRankPolicy> struct histogram_scatter_impl;

template <int RadixBits, class Key>
__device__ __forceinline__ std::uint32_t extract_digit(Key key, int shift) {
    constexpr std::uint32_t kMask =
        static_cast<std::uint32_t>((1u << RadixBits) - 1u);
    return static_cast<std::uint32_t>((key >> shift) & kMask);
}

template <int BlockSize, int ItemsPerThread>
__device__ __forceinline__ std::uint32_t warp_major_tile_index(
    std::uint32_t tile_base, int item) {
    constexpr int kWarpSize = 32;
    const std::uint32_t warp =
        static_cast<std::uint32_t>(threadIdx.x / kWarpSize);
    const std::uint32_t lane =
        static_cast<std::uint32_t>(threadIdx.x & (kWarpSize - 1));
    return tile_base +
           warp * static_cast<std::uint32_t>(kWarpSize * ItemsPerThread) +
           static_cast<std::uint32_t>(item * kWarpSize) + lane;
}

template <int RadixBits, class Key>
__global__ void build_flags_kernel(std::uint32_t* flags, const Key* keys,
                                   std::uint32_t count, int shift) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);
    const std::uint32_t digit = extract_digit<RadixBits>(keys[index], shift);
    for (std::uint32_t bucket = 0; bucket < kNumBuckets; ++bucket) {
        flags[static_cast<std::size_t>(bucket) * count + index] =
            bucket == digit ? 1u : 0u;
    }
}

template <class Key, int BlockSize, int RadixBits>
cudaError_t build_flags(std::uint32_t* flags, const Key* keys,
                        std::uint32_t count, int shift, cudaStream_t stream) {
    if (count == 0)
        return cudaSuccess;
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    build_flags_kernel<RadixBits>
        <<<grid, BlockSize, 0, stream>>>(flags, keys, count, shift);
    return cudaGetLastError();
}

} // namespace algo::cuda::sort::detail
