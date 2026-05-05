#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/common.cuh>

namespace algo::cuda::sort::detail {

template <int BlockSize, int ItemsPerThread, int RadixBits, class Key,
          class OutputLayout>
__global__ void compute_block_histograms_warp_level_multi_split_kernel(
    std::uint32_t* histograms, const Key* keys, std::uint32_t count,
    int shift) {
    static_assert(BlockSize % 32 == 0,
                  "WarpLevelMultiSplitHistogram requires BlockSize to be a multiple "
                  "of the warp size");

    constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);
    constexpr int kWarpSize = 32;
    constexpr int kWarpCount = BlockSize / kWarpSize;
    constexpr unsigned int kWarpMask = 0xffffffffu;

    __shared__ std::uint32_t histogram[kNumBuckets];
    __shared__ std::uint32_t
        warp_counts[static_cast<std::size_t>(kWarpCount) * kNumBuckets];

    for (std::uint32_t index = threadIdx.x;
         index < static_cast<std::uint32_t>(kWarpCount) * kNumBuckets;
         index += blockDim.x) {
        warp_counts[index] = 0;
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x & (kWarpSize - 1));
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const std::uint32_t tile_base =
        blockIdx.x * static_cast<std::uint32_t>(BlockSize * ItemsPerThread);

    for (int item = 0; item < ItemsPerThread; ++item) {
        const std::uint32_t key_index =
            warp_major_tile_index<BlockSize, ItemsPerThread>(tile_base, item);
        const bool valid = key_index < count;
        const std::uint32_t digit =
            valid ? extract_digit<RadixBits>(keys[key_index], shift) : 0u;
        const unsigned int valid_mask = __ballot_sync(kWarpMask, valid);

        unsigned int digit_mask = valid_mask;
        for (int bit = 0; bit < RadixBits; ++bit) {
            const bool bit_set = valid && ((digit & (1u << bit)) != 0u);
            const unsigned int bit_mask = __ballot_sync(kWarpMask, bit_set);
            digit_mask &= ((digit & (1u << bit)) != 0u) ? bit_mask : ~bit_mask;
        }

        if (valid) {
            const int leader_lane = __ffs(digit_mask) - 1;
            if (lane == leader_lane) {
                warp_counts[static_cast<std::size_t>(warp) * kNumBuckets +
                            digit] +=
                    static_cast<std::uint32_t>(__popc(digit_mask));
            }
        }
    }
    __syncthreads();

    for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
         bucket += blockDim.x) {
        std::uint32_t total = 0;
        for (std::uint32_t warp_index = 0; warp_index < kWarpCount;
             ++warp_index) {
            total +=
                warp_counts[static_cast<std::size_t>(warp_index) * kNumBuckets +
                            bucket];
        }
        histogram[bucket] = total;
    }
    __syncthreads();

    const std::uint32_t num_blocks = gridDim.x;
    for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
         bucket += blockDim.x) {
        histograms[OutputLayout::block_histogram_index(
            bucket, blockIdx.x, num_blocks)] = histogram[bucket];
    }
}

template <class Key, int BlockSize, int RadixBits, int ItemsPerThread,
          class OutputLayout>
cudaError_t
compute_block_histograms_warp_level_multi_split(std::uint32_t* histograms,
                                          const Key* keys, std::uint32_t count,
                                          int shift, cudaStream_t stream) {
    if (count == 0)
        return cudaSuccess;
    const auto grid = static_cast<unsigned int>(::algo::ceil_div(
        count, static_cast<std::uint32_t>(BlockSize * ItemsPerThread)));
    compute_block_histograms_warp_level_multi_split_kernel<
        BlockSize, ItemsPerThread, RadixBits, Key, OutputLayout>
        <<<grid, BlockSize, 0, stream>>>(histograms, keys, count, shift);
    return cudaGetLastError();
}

template <> struct histogram_build_impl<WarpLevelMultiSplitHistogram> {
    template <class Key, int BlockSize, int RadixBits, int ItemsPerThread,
              class OutputLayout>
    static cudaError_t run(std::uint32_t* histograms, const Key* input,
                           std::uint32_t count, int shift,
                           cudaStream_t stream) {
        return compute_block_histograms_warp_level_multi_split<
            Key, BlockSize, RadixBits, ItemsPerThread, OutputLayout>(
            histograms, input, count, shift, stream);
    }
};

} // namespace algo::cuda::sort::detail
