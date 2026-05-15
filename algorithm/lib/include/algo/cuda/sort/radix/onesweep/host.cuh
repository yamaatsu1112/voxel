#pragma once

#include <algo/cuda/sort/radix/onesweep/global_offsets.cuh>
#include <algo/cuda/sort/radix/onesweep/partition.cuh>
#include <algo/cuda/sort/value_arrays.cuh>

#include <cstddef>
#include <type_traits>

namespace algo::cuda::sort::detail {

template <class BlockHistogramPolicy, class LocalRankPolicy,
          class GlobalOffsetsBlockHistogramPolicy, int ItemsPerThread,
          int BlockSize, int RadixBits, int KeyBits>
struct sort_impl<
    RadixSort<OneSweepPass<BlockHistogramPolicy, LocalRankPolicy,
                           GlobalOffsetsBlockHistogramPolicy, ItemsPerThread>,
              BlockSize, RadixBits, KeyBits>> {
  static_assert(RadixBits > 0, "RadixSort requires RadixBits > 0");
  static_assert(RadixBits <= 8,
                "OneSweepPass currently requires RadixBits <= 8");
  static_assert(RadixBits < 32, "RadixSort requires RadixBits < 32");
  static_assert(KeyBits > 0, "RadixSort requires KeyBits > 0");
  static_assert(KeyBits % RadixBits == 0,
                "RadixSort requires KeyBits to be a multiple of RadixBits");
  static_assert(BlockSize % 32 == 0,
                "OneSweepPass requires BlockSize "
                "to be a multiple of the warp size");

  using config_type =
      RadixSort<OneSweepPass<BlockHistogramPolicy, LocalRankPolicy,
                             GlobalOffsetsBlockHistogramPolicy, ItemsPerThread>,
                BlockSize, RadixBits, KeyBits>;
  template <class Key>
  using layout_type = workspace_layout<config_type, Key>;
  template <class Key, class... Values>
  using by_key_layout_type =
      by_key_workspace_layout<config_type, Key, value_arrays_t<Values...>>;

  static constexpr std::uint32_t kNumBuckets =
      static_cast<std::uint32_t>(1u << RadixBits);
  static constexpr std::uint32_t kPassCount =
      static_cast<std::uint32_t>(KeyBits / RadixBits);
  static constexpr int kWarpCount = BlockSize / 32;
  static constexpr std::size_t kPartitionSharedBytes =
      sizeof(std::uint32_t) *
      (static_cast<std::size_t>(kNumBuckets) * 3u +
       static_cast<std::size_t>(kWarpCount) * kNumBuckets * 2u);
  static_assert(kPartitionSharedBytes <= 48u * 1024u,
                "OneSweepPass partition scratch memory exceeds the portable "
                "per-block shared memory limit; reduce BlockSize or "
                "RadixBits");

  template <class Key>
  static std::size_t required_workspace_size(std::uint32_t count) {
    static_assert(radix_key_traits<Key>::kSupported,
                  "OneSweepPass key type is not supported");
    static_assert(KeyBits <= radix_key_traits<Key>::kBits,
                  "RadixSort KeyBits exceeds the key type width");
    return layout_type<Key>::required_workspace_size(count);
  }

  template <class Key, class... Values>
  static std::size_t
  required_sort_by_key_workspace_size(std::uint32_t count) {
    static_assert(radix_key_traits<Key>::kSupported,
                  "OneSweepPass key type is not supported");
    static_assert(KeyBits <= radix_key_traits<Key>::kBits,
                  "RadixSort KeyBits exceeds the key type width");
    static_assert(kSupportedValueArrayTypes<Values...>,
                  "sort_by_key requires trivially copyable values");
    return by_key_layout_type<Key, Values...>::required_workspace_size(count);
  }

  template <class Key>
  static cudaError_t sort_keys(Key *d_keys, std::uint32_t count,
                               void *workspace, std::size_t,
                               cudaStream_t stream) {
    static_assert(radix_key_traits<Key>::kSupported,
                  "OneSweepPass key type is not supported");
    static_assert(KeyBits <= radix_key_traits<Key>::kBits,
                  "RadixSort KeyBits exceeds the key type width");
    if (count <= 1)
      return cudaSuccess;
    if (count > kOneSweepLookbackValueMask)
      return cudaErrorInvalidValue;

    auto layout = layout_type<Key>::create(workspace, count);
    cudaError_t status =
        build_onesweep_global_offsets<Key, BlockSize, RadixBits, KeyBits>(
            layout.global_offsets, d_keys, count, stream,
            GlobalOffsetsBlockHistogramPolicy{});
    if (status != cudaSuccess)
      return status;

    Key *input = d_keys;
    Key *output = layout.temp_keys;
    const std::size_t lookback_bytes =
        sizeof(onesweep_lookback_record) *
        static_cast<std::size_t>(kNumBuckets) * layout.num_blocks;

    for (std::uint32_t pass = 0; pass < kPassCount; ++pass) {
      status =
          cudaMemsetAsync(layout.lookback_records, 0, lookback_bytes, stream);
      if (status != cudaSuccess)
        return status;
      status = cudaMemsetAsync(layout.tile_counter, 0, sizeof(std::uint32_t),
                               stream);
      if (status != cudaSuccess)
        return status;
      const int shift =
          static_cast<int>(pass * static_cast<std::uint32_t>(RadixBits));
      const auto grid = static_cast<unsigned int>(layout.num_blocks);
      onesweep_partition_keys_kernel<BlockSize, RadixBits, ItemsPerThread,
                                     BlockHistogramPolicy, LocalRankPolicy>
          <<<grid, BlockSize, 0, stream>>>(
              output, input, layout.global_offsets + pass * kNumBuckets,
              layout.lookback_records, layout.tile_counter, count, shift);
      status = cudaGetLastError();
      if (status != cudaSuccess)
        return status;

      Key *const previous_output = output;
      output = input;
      input = previous_output;
    }

    if (input != d_keys) {
      return ::algo::cuda::copy_buffer<Key, BlockSize>(d_keys, input, count,
                                                       stream);
    }
    return cudaSuccess;
  }

  template <class Key, class... Values>
  static cudaError_t sort_by_key(Key *d_keys, value_arrays_t<Values...> d_values,
                                 std::uint32_t count, void *workspace,
                                 std::size_t, cudaStream_t stream) {
    static_assert(radix_key_traits<Key>::kSupported,
                  "OneSweepPass key type is not supported");
    static_assert(KeyBits <= radix_key_traits<Key>::kBits,
                  "RadixSort KeyBits exceeds the key type width");
    static_assert(kSupportedValueArrayTypes<Values...>,
                  "sort_by_key requires trivially copyable values");
    if (count <= 1)
      return cudaSuccess;
    if (count > kOneSweepLookbackValueMask)
      return cudaErrorInvalidValue;

    auto layout = by_key_layout_type<Key, Values...>::create(workspace, count);
    cudaError_t status =
        build_onesweep_global_offsets<Key, BlockSize, RadixBits, KeyBits>(
            layout.global_offsets, d_keys, count, stream,
            GlobalOffsetsBlockHistogramPolicy{});
    if (status != cudaSuccess)
      return status;

    Key *input_keys = d_keys;
    Key *output_keys = layout.temp_keys;
    auto input_values = d_values;
    auto output_values = layout.temp_values;
    const std::size_t lookback_bytes =
        sizeof(onesweep_lookback_record) *
        static_cast<std::size_t>(kNumBuckets) * layout.num_blocks;

    for (std::uint32_t pass = 0; pass < kPassCount; ++pass) {
      status =
          cudaMemsetAsync(layout.lookback_records, 0, lookback_bytes, stream);
      if (status != cudaSuccess)
        return status;
      status = cudaMemsetAsync(layout.tile_counter, 0, sizeof(std::uint32_t),
                               stream);
      if (status != cudaSuccess)
        return status;
      const int shift =
          static_cast<int>(pass * static_cast<std::uint32_t>(RadixBits));
      const auto grid = static_cast<unsigned int>(layout.num_blocks);
      onesweep_partition_by_key_kernel<BlockSize, RadixBits, ItemsPerThread,
                                       BlockHistogramPolicy, LocalRankPolicy>
          <<<grid, BlockSize, 0, stream>>>(
              output_keys, output_values, input_keys, input_values,
              layout.global_offsets + pass * kNumBuckets,
              layout.lookback_records, layout.tile_counter, count, shift);
      status = cudaGetLastError();
      if (status != cudaSuccess)
        return status;

      Key *const previous_key_output = output_keys;
      output_keys = input_keys;
      input_keys = previous_key_output;

      swap_value_arrays(output_values, input_values);
    }

    if (input_keys != d_keys) {
      status = ::algo::cuda::copy_buffer<Key, BlockSize>(d_keys, input_keys,
                                                         count, stream);
      if (status != cudaSuccess)
        return status;
      return copy_value_array_buffers<BlockSize>(d_values, input_values, count,
                                         stream);
    }
    return cudaSuccess;
  }
};

} // namespace algo::cuda::sort::detail
