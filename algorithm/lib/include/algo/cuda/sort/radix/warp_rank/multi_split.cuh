#pragma once

#include <algo/cuda/sort/radix/common.cuh>

namespace algo::cuda::sort::detail {

template <> struct warp_rank_impl<WarpLevelMultiSplitWarpRank> {
    template <int BlockSize, int RadixBits>
    __device__ static std::uint32_t compute(std::uint32_t* warp_counts,
                                            std::uint32_t digit, bool valid) {
        static_assert(
            BlockSize % 32 == 0,
            "WarpLevelMultiSplitWarpRank requires BlockSize to be a multiple "
            "of the warp size");
        constexpr std::uint32_t kNumBuckets =
            static_cast<std::uint32_t>(1u << RadixBits);
        constexpr int kWarpSize = 32;
        constexpr unsigned int kWarpMask = 0xffffffffu;

        const int lane = static_cast<int>(threadIdx.x & (kWarpSize - 1));
        const int warp = static_cast<int>(threadIdx.x / kWarpSize);
        const unsigned int valid_mask = __ballot_sync(kWarpMask, valid);

        unsigned int digit_mask = valid_mask;
        for (int bit = 0; bit < RadixBits; ++bit) {
            const bool bit_set = valid && ((digit & (1u << bit)) != 0u);
            const unsigned int bit_mask = __ballot_sync(kWarpMask, bit_set);
            digit_mask &= ((digit & (1u << bit)) != 0u) ? bit_mask : ~bit_mask;
        }

        if (!valid) {
            return 0;
        }

        const int leader_lane = __ffs(digit_mask) - 1;
        if (lane == leader_lane) {
            warp_counts[static_cast<std::size_t>(warp) * kNumBuckets + digit] =
                static_cast<std::uint32_t>(__popc(digit_mask));
        }

        const unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
        return static_cast<std::uint32_t>(__popc(digit_mask & lower_lanes));
    }
};

} // namespace algo::cuda::sort::detail
