#pragma once

#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/value_arrays.cuh>

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
    static_assert(KeyBits % RadixBits == 0,
                  "RadixSort requires KeyBits to be a multiple of RadixBits");

    using config_type =
        RadixSort<FlagPrefixSumPass<ScanPolicy>, BlockSize, RadixBits, KeyBits>;
    using flag_scan = flag_scan_impl<ScanPolicy>;
    template <class Key>
    using layout_type = workspace_layout<config_type, Key>;
    template <class Key, class... Values>
    using by_key_layout_type = by_key_workspace_layout<
        config_type, Key, value_arrays_t<Values...>>;

    static constexpr std::uint32_t kKeyBits =
        static_cast<std::uint32_t>(KeyBits);
    static constexpr std::uint32_t kRadixBits =
        static_cast<std::uint32_t>(RadixBits);
    static constexpr std::uint32_t kPassCount =
        kKeyBits / kRadixBits;

    template <class Key>
    static std::size_t required_workspace_size(std::uint32_t count) {
        static_assert(radix_key_traits<Key>::kSupported,
                      "FlagPrefixSumPass key type is not supported");
        static_assert(KeyBits <= radix_key_traits<Key>::kBits,
                      "RadixSort KeyBits exceeds the key type width");
        return layout_type<Key>::required_workspace_size(count);
    }

    template <class Key, class... Values>
    static std::size_t
    required_sort_by_key_workspace_size(std::uint32_t count) {
        static_assert(radix_key_traits<Key>::kSupported,
                      "FlagPrefixSumPass key type is not supported");
        static_assert(KeyBits <= radix_key_traits<Key>::kBits,
                      "RadixSort KeyBits exceeds the key type width");
        static_assert(kSupportedValueArrayTypes<Values...>,
                      "sort_by_key requires trivially copyable values");
        return by_key_layout_type<Key, Values...>::required_workspace_size(
            count);
    }

    template <class Key>
    static cudaError_t sort_keys(Key* d_keys, std::uint32_t count,
                                 void* workspace, std::size_t,
                                 cudaStream_t stream) {
        static_assert(radix_key_traits<Key>::kSupported,
                      "FlagPrefixSumPass key type is not supported");
        static_assert(KeyBits <= radix_key_traits<Key>::kBits,
                      "RadixSort KeyBits exceeds the key type width");
        if (count <= 1)
            return cudaSuccess;

        auto layout = layout_type<Key>::create(workspace, count);

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

    template <class Key, class... Values>
    static cudaError_t sort_by_key(Key* d_keys,
                                   value_arrays_t<Values...> d_values,
                                   std::uint32_t count, void* workspace,
                                   std::size_t, cudaStream_t stream) {
        static_assert(radix_key_traits<Key>::kSupported,
                      "FlagPrefixSumPass key type is not supported");
        static_assert(KeyBits <= radix_key_traits<Key>::kBits,
                      "RadixSort KeyBits exceeds the key type width");
        static_assert(kSupportedValueArrayTypes<Values...>,
                      "sort_by_key requires trivially copyable values");
        if (count <= 1)
            return cudaSuccess;

        auto layout = by_key_layout_type<Key, Values...>::create(workspace,
                                                                 count);

        Key* input_keys = d_keys;
        Key* output_keys = layout.temp_keys;
        auto input_values = d_values;
        auto output_values = layout.temp_values;

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

            status = scatter_impl<ScanPolicy>::template run_by_key<
                Key, BlockSize, RadixBits, decltype(layout), Values...>(
                output_keys, output_values, input_keys, input_values, layout,
                count, shift, stream);
            if (status != cudaSuccess)
                return status;

            Key* const previous_key_output = output_keys;
            output_keys = input_keys;
            input_keys = previous_key_output;

            swap_value_arrays(output_values, input_values);
        }

        if (input_keys != d_keys) {
            cudaError_t status = ::algo::cuda::copy_buffer<Key, BlockSize>(
                d_keys, input_keys, count, stream);
            if (status != cudaSuccess)
                return status;
            return copy_value_array_buffers<BlockSize>(d_values, input_values, count,
                                               stream);
        }
        return cudaSuccess;
    }
};

} // namespace algo::cuda::sort::detail
