#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/radix/flag_prefix_sum/flattened/kernels.cuh>

#include <limits>

namespace algo::cuda::sort::detail {

template <int BlockSize, int RadixBits, int KeyBits>
struct workspace_layout<RadixSort<FlagPrefixSumPass<FlattenedScan>, BlockSize,
                                  RadixBits, KeyBits>> {
    static constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);

    std::uint32_t* temp_keys = nullptr;
    std::uint32_t* flags = nullptr;
    void* scan_workspace = nullptr;
    std::size_t scan_workspace_size = 0;

    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= 1)
            return 0;
        const std::size_t flags_count =
            static_cast<std::size_t>(kNumBuckets) * count;
        const std::size_t scan_workspace_bytes =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(
                static_cast<std::uint32_t>(flags_count));

        std::size_t offset = 0;

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * flags_count;

        offset = align_up<std::max_align_t>(offset);
        offset += scan_workspace_bytes;

        return offset;
    }

    static workspace_layout create(void* workspace, std::uint32_t count) {
        workspace_layout layout{};
        const std::size_t flags_count =
            static_cast<std::size_t>(kNumBuckets) * count;
        const std::size_t scan_workspace_bytes =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(
                static_cast<std::uint32_t>(flags_count));

        std::size_t offset = 0;

        offset = align_up<std::uint32_t>(offset);
        layout.temp_keys = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        layout.flags = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * flags_count;

        offset = align_up<std::max_align_t>(offset);
        layout.scan_workspace = static_cast<std::byte*>(workspace) + offset;
        layout.scan_workspace_size = scan_workspace_bytes;

        return layout;
    }
};

template <int BlockSize, int RadixBits, int KeyBits, class Value>
struct pair_workspace_layout<
    RadixSort<FlagPrefixSumPass<FlattenedScan>, BlockSize, RadixBits, KeyBits>,
    Value> {
    static constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);

    std::uint32_t* temp_keys = nullptr;
    Value* temp_values = nullptr;
    std::uint32_t* flags = nullptr;
    void* scan_workspace = nullptr;
    std::size_t scan_workspace_size = 0;

    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= 1)
            return 0;
        const std::size_t flags_count =
            static_cast<std::size_t>(kNumBuckets) * count;
        const std::size_t scan_workspace_bytes =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(
                static_cast<std::uint32_t>(flags_count));

        std::size_t offset = 0;

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

        offset = align_up<Value>(offset);
        offset += sizeof(Value) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * flags_count;

        offset = align_up<std::max_align_t>(offset);
        offset += scan_workspace_bytes;

        return offset;
    }

    static pair_workspace_layout create(void* workspace, std::uint32_t count) {
        pair_workspace_layout layout{};
        const std::size_t flags_count =
            static_cast<std::size_t>(kNumBuckets) * count;
        const std::size_t scan_workspace_bytes =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(
                static_cast<std::uint32_t>(flags_count));

        std::size_t offset = 0;

        offset = align_up<std::uint32_t>(offset);
        layout.temp_keys = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

        offset = align_up<Value>(offset);
        layout.temp_values = pointer_at<Value>(workspace, offset);
        offset += sizeof(Value) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        layout.flags = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * flags_count;

        offset = align_up<std::max_align_t>(offset);
        layout.scan_workspace = static_cast<std::byte*>(workspace) + offset;
        layout.scan_workspace_size = scan_workspace_bytes;

        return layout;
    }
};

template <> struct flag_scan_impl<FlattenedScan> {
    template <int BlockSize, int RadixBits, class Layout, class Key>
    static cudaError_t scan_flags(Layout& layout, Key*, std::uint32_t count,
                                  int, cudaStream_t stream) {
        constexpr std::uint32_t kNumBuckets =
            static_cast<std::uint32_t>(1u << RadixBits);
        if (count > std::numeric_limits<std::uint32_t>::max() / kNumBuckets) {
            return cudaErrorInvalidValue;
        }
        const std::uint32_t flattened_count = kNumBuckets * count;
        return algo::cuda::scan::exclusive_sum(
            layout.flags, flattened_count, layout.scan_workspace,
            layout.scan_workspace_size, stream);
    }
};

template <> struct scatter_impl<FlattenedScan> {
    template <class Key, int BlockSize, int RadixBits, class Layout>
    static cudaError_t run(Key* output, const Key* input, const Layout& layout,
                           std::uint32_t count, int shift,
                           cudaStream_t stream) {
        return scatter_keys_flattened<Key, BlockSize, RadixBits>(
            output, input, layout.flags, count, shift, stream);
    }

    template <class Key, class Value, int BlockSize, int RadixBits,
              class Layout>
    static cudaError_t
    run_pairs(Key* output_keys, Value* output_values, const Key* input_keys,
              const Value* input_values, const Layout& layout,
              std::uint32_t count, int shift, cudaStream_t stream) {
        return scatter_pairs_flattened<Key, Value, BlockSize, RadixBits>(
            output_keys, output_values, input_keys, input_values, layout.flags,
            count, shift, stream);
    }
};

} // namespace algo::cuda::sort::detail
