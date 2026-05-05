#pragma once

#include <algo/cuda/sort/radix/common.cuh>

#include <limits>

namespace algo::cuda::sort::detail {

struct histogram_scan {
    template <int BlockSize, int RadixBits, class Layout>
    static cudaError_t scan_histograms(Layout& layout, std::uint32_t,
                                       cudaStream_t stream) {
        constexpr std::uint32_t kNumBuckets =
            static_cast<std::uint32_t>(1u << RadixBits);
        if (layout.num_blocks >
            std::numeric_limits<std::uint32_t>::max() / kNumBuckets) {
            return cudaErrorInvalidValue;
        }
        const std::uint32_t flattened_count = kNumBuckets * layout.num_blocks;
        return algo::cuda::scan::exclusive_sum(
            layout.histograms, flattened_count, layout.scan_workspace,
            layout.scan_workspace_size, stream);
    }
};

} // namespace algo::cuda::sort::detail
