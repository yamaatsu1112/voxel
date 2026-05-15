#pragma once

#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/radix/warp_rank/count_update.cuh>

namespace algo::cuda::sort::detail {

template <> struct warp_rank_impl<BucketBallotWarpRank> {
    template <class CountUpdater, int BlockSize, int RadixBits>
    __device__ static std::uint32_t compute(std::uint32_t* warp_counts,
                                            std::uint32_t digit, bool valid) {
        static_assert(
            BlockSize % 32 == 0,
            "BucketBallotWarpRank requires BlockSize to be a multiple "
            "of the warp size");
        constexpr std::uint32_t kNumBuckets =
            static_cast<std::uint32_t>(1u << RadixBits);
        constexpr int kWarpSize = 32;
        constexpr unsigned int kWarpMask = 0xffffffffu;

        const int lane = static_cast<int>(threadIdx.x & (kWarpSize - 1));
        const int warp = static_cast<int>(threadIdx.x / kWarpSize);

        std::uint32_t rank_in_warp = 0;
        for (std::uint32_t bucket = 0; bucket < kNumBuckets; ++bucket) {
            const unsigned int bucket_mask =
                __ballot_sync(kWarpMask, valid && digit == bucket);
            if (lane == 0) {
                const std::size_t index =
                    static_cast<std::size_t>(warp) * kNumBuckets + bucket;
                const auto count =
                    static_cast<std::uint32_t>(__popc(bucket_mask));
                CountUpdater::update(warp_counts, index, count);
            }
            if (valid && digit == bucket) {
                const unsigned int lower_lanes =
                    lane == 0 ? 0u : ((1u << lane) - 1u);
                rank_in_warp = static_cast<std::uint32_t>(
                    __popc(bucket_mask & lower_lanes));
            }
        }
        return rank_in_warp;
    }
};

} // namespace algo::cuda::sort::detail
