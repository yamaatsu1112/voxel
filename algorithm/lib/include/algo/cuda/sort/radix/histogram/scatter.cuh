#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/radix/histogram/layout.cuh>
#include <algo/cuda/sort/radix/local_rank.cuh>
#include <algo/cuda/sort/value_arrays.cuh>

namespace algo::cuda::sort::detail {

template <int BlockSize, int RadixBits, int ItemsPerThread,
          class LocalRankPolicy, class Key>
__global__ void scatter_keys_histogram_kernel(
    Key* output, const Key* input, const std::uint32_t* scanned_histograms,
    std::uint32_t count, int shift) {
    using local_rank = histogram_local_rank_impl<LocalRankPolicy>;
    constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);
    constexpr int kWarpSize = 32;
    constexpr int kWarpCount = BlockSize / kWarpSize;

    __shared__ std::uint32_t block_histogram[kNumBuckets];
    __shared__ std::uint32_t
        warp_counts[static_cast<std::size_t>(kWarpCount) * kNumBuckets];
    __shared__ std::uint32_t
        warp_bucket_scratch[static_cast<std::size_t>(kWarpCount) * kNumBuckets];

    Key keys[ItemsPerThread];
    std::uint32_t digits[ItemsPerThread];
    std::uint32_t ranks[ItemsPerThread];
    bool valid_items[ItemsPerThread];
    const std::uint32_t tile_base =
        blockIdx.x * static_cast<std::uint32_t>(BlockSize * ItemsPerThread);
    for (int item = 0; item < ItemsPerThread; ++item) {
        const std::uint32_t global_index =
            warp_major_tile_index<BlockSize, ItemsPerThread>(tile_base, item);
        const bool valid = global_index < count;
        const Key key = valid ? input[global_index] : Key{};
        keys[item] = key;
        digits[item] = valid ? extract_digit<RadixBits>(key, shift) : 0u;
        ranks[item] = 0;
        valid_items[item] = valid;
    }

    local_rank::template compute<BlockSize, RadixBits, ItemsPerThread>(
        ranks, block_histogram, digits, valid_items, warp_counts,
        warp_bucket_scratch);

    const std::uint32_t num_blocks = gridDim.x;
    for (int item = 0; item < ItemsPerThread; ++item) {
        if (valid_items[item]) {
            const std::uint32_t digit = digits[item];
            const std::uint32_t block_bucket_base =
                scanned_histograms
                    [histogram_layout::block_histogram_index(
                        digit, blockIdx.x, num_blocks)];
            output[block_bucket_base + ranks[item]] = keys[item];
        }
    }
}

template <class Key, int BlockSize, int RadixBits, int ItemsPerThread,
          class LocalRankPolicy>
cudaError_t scatter_keys_histogram(
    Key* output, const Key* input, const std::uint32_t* scanned_histograms,
    std::uint32_t count, int shift, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto grid = static_cast<unsigned int>(::algo::ceil_div(
        count, static_cast<std::uint32_t>(BlockSize * ItemsPerThread)));
    scatter_keys_histogram_kernel<BlockSize, RadixBits, ItemsPerThread,
                                  LocalRankPolicy>
        <<<grid, BlockSize, 0, stream>>>(output, input, scanned_histograms,
                                         count, shift);
    return cudaGetLastError();
}

template <int BlockSize, int RadixBits, int ItemsPerThread,
          class LocalRankPolicy, class Key, class... Values>
__global__ void scatter_by_key_histogram_kernel(
    Key* output_keys, value_arrays_t<Values...> output_values,
    const Key* input_keys, value_arrays_t<Values...> input_values,
    const std::uint32_t* scanned_histograms, std::uint32_t count, int shift) {
    using local_rank = histogram_local_rank_impl<LocalRankPolicy>;
    constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);
    constexpr int kWarpSize = 32;
    constexpr int kWarpCount = BlockSize / kWarpSize;

    __shared__ std::uint32_t block_histogram[kNumBuckets];
    __shared__ std::uint32_t
        warp_counts[static_cast<std::size_t>(kWarpCount) * kNumBuckets];
    __shared__ std::uint32_t
        warp_bucket_scratch[static_cast<std::size_t>(kWarpCount) * kNumBuckets];

    Key keys[ItemsPerThread];
    std::uint32_t digits[ItemsPerThread];
    std::uint32_t ranks[ItemsPerThread];
    bool valid_items[ItemsPerThread];
    const std::uint32_t tile_base =
        blockIdx.x * static_cast<std::uint32_t>(BlockSize * ItemsPerThread);
    for (int item = 0; item < ItemsPerThread; ++item) {
        const std::uint32_t global_index =
            warp_major_tile_index<BlockSize, ItemsPerThread>(tile_base, item);
        const bool valid = global_index < count;
        const Key key = valid ? input_keys[global_index] : Key{};
        keys[item] = key;
        digits[item] = valid ? extract_digit<RadixBits>(key, shift) : 0u;
        ranks[item] = 0;
        valid_items[item] = valid;
    }

    local_rank::template compute<BlockSize, RadixBits, ItemsPerThread>(
        ranks, block_histogram, digits, valid_items, warp_counts,
        warp_bucket_scratch);

    const std::uint32_t num_blocks = gridDim.x;
    for (int item = 0; item < ItemsPerThread; ++item) {
        if (valid_items[item]) {
            const std::uint32_t digit = digits[item];
            const std::uint32_t block_bucket_base =
                scanned_histograms
                    [histogram_layout::block_histogram_index(
                        digit, blockIdx.x, num_blocks)];
            const std::uint32_t output_index =
                block_bucket_base + ranks[item];
            const std::uint32_t input_index =
                warp_major_tile_index<BlockSize, ItemsPerThread>(tile_base,
                                                                 item);
            output_keys[output_index] = keys[item];
            copy_value_array_item(output_values, output_index, input_values,
                          input_index);
        }
    }
}

template <class Key, int BlockSize, int RadixBits, int ItemsPerThread,
          class LocalRankPolicy, class... Values>
cudaError_t scatter_by_key_histogram(
    Key* output_keys, value_arrays_t<Values...> output_values,
    const Key* input_keys, value_arrays_t<Values...> input_values,
    const std::uint32_t* scanned_histograms, std::uint32_t count, int shift,
    cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto grid = static_cast<unsigned int>(::algo::ceil_div(
        count, static_cast<std::uint32_t>(BlockSize * ItemsPerThread)));
    scatter_by_key_histogram_kernel<BlockSize, RadixBits, ItemsPerThread,
                                    LocalRankPolicy>
        <<<grid, BlockSize, 0, stream>>>(output_keys, output_values,
                                         input_keys, input_values,
                                         scanned_histograms, count, shift);
    return cudaGetLastError();
}

template <class LocalRankPolicy>
struct histogram_scatter_impl {
    template <class Key, int BlockSize, int RadixBits, int ItemsPerThread,
              class Layout>
    static cudaError_t run(Key* output, const Key* input, const Layout& layout,
                           std::uint32_t count, int shift,
                           cudaStream_t stream) {
        return scatter_keys_histogram<Key, BlockSize, RadixBits,
                                                ItemsPerThread,
                                                LocalRankPolicy>(
            output, input, layout.histograms, count, shift, stream);
    }

    template <class Key, int BlockSize, int RadixBits, int ItemsPerThread,
              class Layout, class... Values>
    static cudaError_t
    run_by_key(Key* output_keys, value_arrays_t<Values...> output_values,
               const Key* input_keys, value_arrays_t<Values...> input_values,
               const Layout& layout, std::uint32_t count, int shift,
               cudaStream_t stream) {
        return scatter_by_key_histogram<Key, BlockSize, RadixBits,
                                        ItemsPerThread, LocalRankPolicy,
                                        Values...>(
            output_keys, output_values, input_keys, input_values,
            layout.histograms, count, shift, stream);
    }
};

} // namespace algo::cuda::sort::detail
