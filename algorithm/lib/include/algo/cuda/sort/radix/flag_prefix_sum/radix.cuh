#pragma once

#include <algo/cuda/sort/radix/common.cuh>

namespace algo::cuda::sort::detail {

template <class Config> struct sort_impl;

template <class ScanPolicy, int BlockSize, int RadixBits, int KeyBits>
struct sort_impl<
    RadixSort<FlagPrefixSumPass<ScanPolicy>, BlockSize, RadixBits, KeyBits>> {
    static_assert(RadixBits > 0, "RadixSort requires RadixBits > 0");
    static_assert(RadixBits <= 5,
                  "FlagPrefixSumPass currently requires RadixBits <= 5");
    static_assert(RadixBits < 32, "RadixSort requires RadixBits < 32");
    static_assert(KeyBits > 0, "RadixSort requires KeyBits > 0");
    static_assert(KeyBits <= 32, "RadixSort currently requires KeyBits <= 32");
    static_assert(KeyBits % RadixBits == 0,
                  "RadixSort requires KeyBits to be a multiple of RadixBits");

    using config_type =
        RadixSort<FlagPrefixSumPass<ScanPolicy>, BlockSize, RadixBits, KeyBits>;
    using flag_scan = flag_scan_impl<ScanPolicy>;
    using layout_type = workspace_layout<config_type>;
    template <class Value>
    using pair_layout_type = pair_workspace_layout<config_type, Value>;

    static constexpr std::uint32_t kKeyBits =
        static_cast<std::uint32_t>(KeyBits);
    static constexpr std::uint32_t kRadixBits =
        static_cast<std::uint32_t>(RadixBits);
    static constexpr std::uint32_t kPassCount =
        kKeyBits / kRadixBits;

    template <class Key>
    static std::size_t required_workspace_size(std::uint32_t count) {
        static_assert(std::is_same_v<Key, std::uint32_t>,
                      "FlagPrefixSumPass currently only supports uint32_t");
        static_assert(KeyBits <= static_cast<int>(sizeof(Key) * 8u),
                      "RadixSort KeyBits exceeds the key type width");
        return layout_type::required_workspace_size(count);
    }

    template <class Key, class Value>
    static std::size_t required_pairs_workspace_size(std::uint32_t count) {
        static_assert(std::is_same_v<Key, std::uint32_t>,
                      "FlagPrefixSumPass currently only supports uint32_t");
        static_assert(KeyBits <= static_cast<int>(sizeof(Key) * 8u),
                      "RadixSort KeyBits exceeds the key type width");
        static_assert(std::is_trivially_copyable_v<Value>,
                      "sort_pairs requires trivially copyable values");
        return pair_layout_type<Value>::required_workspace_size(count);
    }

    template <class Key>
    static cudaError_t sort_keys(Key* d_keys, std::uint32_t count,
                                 void* workspace, std::size_t,
                                 cudaStream_t stream) {
        static_assert(std::is_same_v<Key, std::uint32_t>,
                      "FlagPrefixSumPass currently only supports uint32_t");
        static_assert(KeyBits <= static_cast<int>(sizeof(Key) * 8u),
                      "RadixSort KeyBits exceeds the key type width");
        if (count <= 1)
            return cudaSuccess;

        auto layout = layout_type::create(workspace, count);

        Key* input = d_keys;
        Key* output = layout.temp_keys;

        for (std::uint32_t pass = 0; pass < kPassCount; ++pass) {
            const int shift =
                static_cast<int>(pass * static_cast<std::uint32_t>(RadixBits));

            cudaError_t status = build_flags<Key, BlockSize, RadixBits>(
                layout.flags, input, count, shift, stream);
            if (status != cudaSuccess)
                return status;

            status = flag_scan::template scan_flags<BlockSize, RadixBits>(
                layout, input, count, shift, stream);
            if (status != cudaSuccess)
                return status;

            status = scatter_impl<ScanPolicy>::template run<Key, BlockSize,
                                                            RadixBits>(
                output, input, layout, count, shift, stream);
            if (status != cudaSuccess)
                return status;

            Key* const previous_output = output;
            output = input;
            input = previous_output;
        }

        if (input != d_keys) {
            return ::algo::cuda::copy_buffer<Key, BlockSize>(d_keys, input,
                                                             count, stream);
        }
        return cudaSuccess;
    }

    template <class Key, class Value>
    static cudaError_t sort_pairs(Key* d_keys, Value* d_values,
                                  std::uint32_t count, void* workspace,
                                  std::size_t, cudaStream_t stream) {
        static_assert(std::is_same_v<Key, std::uint32_t>,
                      "FlagPrefixSumPass currently only supports uint32_t");
        static_assert(KeyBits <= static_cast<int>(sizeof(Key) * 8u),
                      "RadixSort KeyBits exceeds the key type width");
        static_assert(std::is_trivially_copyable_v<Value>,
                      "sort_pairs requires trivially copyable values");
        if (count <= 1)
            return cudaSuccess;

        auto layout = pair_layout_type<Value>::create(workspace, count);

        Key* input_keys = d_keys;
        Key* output_keys = layout.temp_keys;
        Value* input_values = d_values;
        Value* output_values = layout.temp_values;

        for (std::uint32_t pass = 0; pass < kPassCount; ++pass) {
            const int shift =
                static_cast<int>(pass * static_cast<std::uint32_t>(RadixBits));

            cudaError_t status = build_flags<Key, BlockSize, RadixBits>(
                layout.flags, input_keys, count, shift, stream);
            if (status != cudaSuccess)
                return status;

            status = flag_scan::template scan_flags<BlockSize, RadixBits>(
                layout, input_keys, count, shift, stream);
            if (status != cudaSuccess)
                return status;

            status = scatter_impl<ScanPolicy>::template run_pairs<
                Key, Value, BlockSize, RadixBits>(
                output_keys, output_values, input_keys, input_values, layout,
                count, shift, stream);
            if (status != cudaSuccess)
                return status;

            Key* const previous_key_output = output_keys;
            output_keys = input_keys;
            input_keys = previous_key_output;

            Value* const previous_value_output = output_values;
            output_values = input_values;
            input_values = previous_value_output;
        }

        if (input_keys != d_keys) {
            cudaError_t status = ::algo::cuda::copy_buffer<Key, BlockSize>(
                d_keys, input_keys, count, stream);
            if (status != cudaSuccess)
                return status;
            return ::algo::cuda::copy_buffer<Value, BlockSize>(
                d_values, input_values, count, stream);
        }
        return cudaSuccess;
    }
};

} // namespace algo::cuda::sort::detail
