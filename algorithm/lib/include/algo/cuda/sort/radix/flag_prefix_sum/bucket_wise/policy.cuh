#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/flag_prefix_sum/bucket_wise/kernels.cuh>
#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/value_arrays.cuh>

namespace algo::cuda::sort::detail {

template <int BlockSize, int RadixBits, int KeyBits, class Key>
struct workspace_layout<RadixSort<FlagPrefixSumPass<BucketWiseScan>, BlockSize,
                                  RadixBits, KeyBits>,
                        Key> {
    static constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);

    Key* temp_keys = nullptr;
    std::uint32_t* flags = nullptr;
    std::uint32_t* bucket_offsets = nullptr;
    void* scan_workspace = nullptr;
    std::size_t scan_workspace_size = 0;

    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= 1)
            return 0;
        const std::size_t flags_count =
            static_cast<std::size_t>(kNumBuckets) * count;
        const std::size_t scan_workspace_per_bucket =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(count);

        std::size_t offset = 0;

        offset = align_up<Key>(offset);
        offset += sizeof(Key) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * flags_count;

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(kNumBuckets);

        offset = align_up<std::max_align_t>(offset);
        offset += scan_workspace_per_bucket;

        return offset;
    }

    static workspace_layout create(void* workspace, std::uint32_t count) {
        workspace_layout layout{};
        const std::size_t scan_workspace_per_bucket =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(count);

        std::size_t offset = 0;

        offset = align_up<Key>(offset);
        layout.temp_keys = pointer_at<Key>(workspace, offset);
        offset += sizeof(Key) * static_cast<std::size_t>(count);

        offset = align_up<std::uint32_t>(offset);
        layout.flags = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) *
                  (static_cast<std::size_t>(kNumBuckets) * count);

        offset = align_up<std::uint32_t>(offset);
        layout.bucket_offsets = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(kNumBuckets);

        offset = align_up<std::max_align_t>(offset);
        layout.scan_workspace = static_cast<std::byte*>(workspace) + offset;
        layout.scan_workspace_size = scan_workspace_per_bucket;

        return layout;
    }
};

template <int BlockSize, int RadixBits, int KeyBits, class Key,
          class... Values>
struct by_key_workspace_layout<
    RadixSort<FlagPrefixSumPass<BucketWiseScan>, BlockSize, RadixBits, KeyBits>,
    Key,
    value_arrays_t<Values...>> {
    static constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);

    Key* temp_keys = nullptr;
    value_arrays_t<Values...> temp_values{};
    std::uint32_t* flags = nullptr;
    std::uint32_t* bucket_offsets = nullptr;
    void* scan_workspace = nullptr;
    std::size_t scan_workspace_size = 0;

    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= 1)
            return 0;
        const std::size_t flags_count =
            static_cast<std::size_t>(kNumBuckets) * count;
        const std::size_t scan_workspace_per_bucket =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(count);

        std::size_t offset = 0;

        offset = align_up<Key>(offset);
        offset += sizeof(Key) * static_cast<std::size_t>(count);

        offset = add_value_array_temp_storage_size<Values...>(offset, count);

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * flags_count;

        offset = align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(kNumBuckets);

        offset = align_up<std::max_align_t>(offset);
        offset += scan_workspace_per_bucket;

        return offset;
    }

    static by_key_workspace_layout create(void* workspace,
                                          std::uint32_t count) {
        by_key_workspace_layout layout{};
        const std::size_t scan_workspace_per_bucket =
            algo::cuda::scan::required_workspace_size<std::uint32_t>(count);

        std::size_t offset = 0;

        offset = align_up<Key>(offset);
        layout.temp_keys = pointer_at<Key>(workspace, offset);
        offset += sizeof(Key) * static_cast<std::size_t>(count);

        layout.temp_values =
            allocate_temp_value_arrays<Values...>(workspace, offset, count);

        offset = align_up<std::uint32_t>(offset);
        layout.flags = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) *
                  (static_cast<std::size_t>(kNumBuckets) * count);

        offset = align_up<std::uint32_t>(offset);
        layout.bucket_offsets = pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t) * static_cast<std::size_t>(kNumBuckets);

        offset = align_up<std::max_align_t>(offset);
        layout.scan_workspace = static_cast<std::byte*>(workspace) + offset;
        layout.scan_workspace_size = scan_workspace_per_bucket;

        return layout;
    }
};

template <> struct flag_scan_impl<BucketWiseScan> {
    template <int BlockSize, int RadixBits, class Layout, class Key>
    static cudaError_t scan_flags(Layout& layout, Key* input,
                                  std::uint32_t count, int shift,
                                  cudaStream_t stream) {
        constexpr std::uint32_t kNumBuckets =
            static_cast<std::uint32_t>(1u << RadixBits);

        for (std::uint32_t bucket = 0; bucket < kNumBuckets; ++bucket) {
            auto* bucket_flags =
                layout.flags + static_cast<std::size_t>(bucket) * count;
            cudaError_t status = algo::cuda::scan::exclusive_sum(
                bucket_flags, count, layout.scan_workspace,
                layout.scan_workspace_size, stream);
            if (status != cudaSuccess)
                return status;
        }

        return compute_bucket_offsets<Key, RadixBits>(
            layout.bucket_offsets, layout.flags, input, count, shift, stream);
    }
};

template <> struct scatter_impl<BucketWiseScan> {
    template <class Key, int BlockSize, int RadixBits, class Layout>
    static cudaError_t run(Key* output, const Key* input, const Layout& layout,
                           std::uint32_t count, int shift,
                           cudaStream_t stream) {
        return scatter_keys<Key, BlockSize, RadixBits>(
            output, input, layout.flags, layout.bucket_offsets, count, shift,
            stream);
    }

    template <class Key, int BlockSize, int RadixBits, class Layout,
              class... Values>
    static cudaError_t
    run_by_key(Key* output_keys, value_arrays_t<Values...> output_values,
               const Key* input_keys, value_arrays_t<Values...> input_values,
               const Layout& layout, std::uint32_t count, int shift,
               cudaStream_t stream) {
        return scatter_by_key<Key, BlockSize, RadixBits, Values...>(
            output_keys, output_values, input_keys, input_values, layout.flags,
            layout.bucket_offsets, count, shift, stream);
    }
};

} // namespace algo::cuda::sort::detail
