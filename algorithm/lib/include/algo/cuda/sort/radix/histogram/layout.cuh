#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/common.cuh>

namespace algo::cuda::sort::detail {

struct histogram_layout {
    __host__ __device__ static std::size_t
    block_histogram_index(std::uint32_t bucket, std::uint32_t block,
                          std::uint32_t num_blocks) {
        return static_cast<std::size_t>(bucket) * num_blocks + block;
    }
};

template <class HistogramPolicy, class LocalRankPolicy, int ItemsPerThread,
          int BlockSize, int RadixBits, int KeyBits>
struct workspace_layout<
    RadixSort<HistogramPass<HistogramPolicy,
                            LocalRankPolicy, ItemsPerThread>,
              BlockSize, RadixBits, KeyBits>> {
    static constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);

    std::uint32_t* temp_keys = nullptr;
    std::uint32_t* histograms = nullptr;
    void* scan_workspace = nullptr;
    std::size_t scan_workspace_size = 0;
    std::uint32_t num_blocks = 0;

    static std::uint32_t block_count(std::uint32_t count) {
        return ::algo::ceil_div(
            count, static_cast<std::uint32_t>(BlockSize * ItemsPerThread));
    }

    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= 1) return 0;

        const std::uint32_t blocks = block_count(count);
        const std::size_t histogram_count =
            static_cast<std::size_t>(kNumBuckets) * blocks;
        const std::size_t scan_workspace_bytes =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(
                static_cast<std::uint32_t>(histogram_count));

        std::size_t offset = 0;

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * histogram_count;

        offset = align_up<std::max_align_t>(offset);
        offset += scan_workspace_bytes;

        return offset;
    }

    static workspace_layout create(void* workspace, std::uint32_t count) {
        workspace_layout layout{};
        layout.num_blocks = block_count(count);
        const std::size_t histogram_count =
            static_cast<std::size_t>(kNumBuckets) * layout.num_blocks;
        const std::size_t scan_workspace_bytes =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(
                static_cast<std::uint32_t>(histogram_count));

        std::size_t offset = 0;

        offset = align_up<std::uint32_t>(offset);
        layout.temp_keys = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        layout.histograms = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * histogram_count;

        offset = align_up<std::max_align_t>(offset);
        layout.scan_workspace = static_cast<std::byte*>(workspace) + offset;
        layout.scan_workspace_size = scan_workspace_bytes;

        return layout;
    }
};

template <class HistogramPolicy, class LocalRankPolicy, int ItemsPerThread,
          int BlockSize, int RadixBits, int KeyBits, class Value>
struct pair_workspace_layout<
    RadixSort<HistogramPass<HistogramPolicy,
                            LocalRankPolicy, ItemsPerThread>,
              BlockSize, RadixBits, KeyBits>,
    Value> {
    static constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);

    std::uint32_t* temp_keys = nullptr;
    Value* temp_values = nullptr;
    std::uint32_t* histograms = nullptr;
    void* scan_workspace = nullptr;
    std::size_t scan_workspace_size = 0;
    std::uint32_t num_blocks = 0;

    static std::uint32_t block_count(std::uint32_t count) {
        return ::algo::ceil_div(
            count, static_cast<std::uint32_t>(BlockSize * ItemsPerThread));
    }

    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= 1) return 0;

        const std::uint32_t blocks = block_count(count);
        const std::size_t histogram_count =
            static_cast<std::size_t>(kNumBuckets) * blocks;
        const std::size_t scan_workspace_bytes =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(
                static_cast<std::uint32_t>(histogram_count));

        std::size_t offset = 0;

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

        offset = align_up<Value>(offset);
        offset += sizeof(Value) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * histogram_count;

        offset = align_up<std::max_align_t>(offset);
        offset += scan_workspace_bytes;

        return offset;
    }

    static pair_workspace_layout create(void* workspace, std::uint32_t count) {
        pair_workspace_layout layout{};
        layout.num_blocks = block_count(count);
        const std::size_t histogram_count =
            static_cast<std::size_t>(kNumBuckets) * layout.num_blocks;
        const std::size_t scan_workspace_bytes =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(
                static_cast<std::uint32_t>(histogram_count));

        std::size_t offset = 0;

        offset = align_up<std::uint32_t>(offset);
        layout.temp_keys = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

        offset = align_up<Value>(offset);
        layout.temp_values = pointer_at<Value>(workspace, offset);
        offset += sizeof(Value) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        layout.histograms = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * histogram_count;

        offset = align_up<std::max_align_t>(offset);
        layout.scan_workspace = static_cast<std::byte*>(workspace) + offset;
        layout.scan_workspace_size = scan_workspace_bytes;

        return layout;
    }
};

} // namespace algo::cuda::sort::detail
