#pragma once

#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/radix/warp_rank/bucket_ballot.cuh>
#include <algo/cuda/sort/radix/warp_rank/multi_split.cuh>

namespace algo::cuda::sort::detail {

template <class WarpRankPolicy> struct histogram_local_rank_impl {
    template <int BlockSize, int RadixBits, int ItemsPerThread>
    __device__ static void compute(
        std::uint32_t (&ranks)[ItemsPerThread],
        std::uint32_t* block_histogram,
        std::uint32_t (&digits)[ItemsPerThread],
        bool (&valid_items)[ItemsPerThread], std::uint32_t* warp_counts,
        std::uint32_t* warp_bucket_scratch) {
        static_assert(
            BlockSize % 32 == 0,
            "histogram local rank requires BlockSize to be a multiple "
            "of the warp size");
        static_assert(ItemsPerThread > 0,
                      "histogram local rank requires ItemsPerThread > 0");

        constexpr std::uint32_t kNumBuckets =
            static_cast<std::uint32_t>(1u << RadixBits);
        constexpr int kWarpSize = 32;
        constexpr int kWarpCount = BlockSize / kWarpSize;

        std::uint32_t item_prefix_in_warp[ItemsPerThread];

        for (std::uint32_t index = threadIdx.x;
             index < static_cast<std::uint32_t>(kWarpCount) * kNumBuckets;
             index += blockDim.x) {
            warp_counts[index] = 0;
        }
        for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
             bucket += blockDim.x) {
            block_histogram[bucket] = 0;
        }
        __syncthreads();

        const std::uint32_t warp =
            static_cast<std::uint32_t>(threadIdx.x / kWarpSize);
        for (int item = 0; item < ItemsPerThread; ++item) {
            const std::uint32_t digit = digits[item];
            item_prefix_in_warp[item] =
                warp_counts[static_cast<std::size_t>(warp) * kNumBuckets +
                            digit];

            ranks[item] =
                warp_rank_impl<WarpRankPolicy>::template compute<
                    warp_rank_count_add, BlockSize, RadixBits>(
                    warp_counts, digit, valid_items[item]);
            __syncthreads();
        }

        for (std::uint32_t index = threadIdx.x;
             index < static_cast<std::uint32_t>(kWarpCount) * kNumBuckets;
             index += blockDim.x) {
            const std::uint32_t warp_index = index / kNumBuckets;
            const std::uint32_t bucket = index % kNumBuckets;
            std::uint32_t offset = 0;
            for (std::uint32_t previous_warp = 0; previous_warp < warp_index;
                 ++previous_warp) {
                offset += warp_counts[static_cast<std::size_t>(previous_warp) *
                                          kNumBuckets +
                                      bucket];
            }
            warp_bucket_scratch[index] = offset;
        }
        __syncthreads();

        for (int item = 0; item < ItemsPerThread; ++item) {
            if (valid_items[item]) {
                const std::uint32_t digit = digits[item];
                ranks[item] +=
                    warp_bucket_scratch[static_cast<std::size_t>(warp) *
                                            kNumBuckets +
                                        digit] +
                    item_prefix_in_warp[item];
            } else {
                ranks[item] = 0;
            }
        }

        for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
             bucket += blockDim.x) {
            std::uint32_t total = 0;
            for (std::uint32_t warp_index = 0; warp_index < kWarpCount;
                 ++warp_index) {
                total +=
                    warp_counts[static_cast<std::size_t>(warp_index) *
                                    kNumBuckets +
                                bucket];
            }
            block_histogram[bucket] = total;
        }
        __syncthreads();
    }
};

} // namespace algo::cuda::sort::detail
