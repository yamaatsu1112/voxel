#pragma once

#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/radix/local_rank.cuh>
#include <algo/cuda/sort/radix/onesweep/lookback.cuh>

namespace algo::cuda::sort::detail {

template <class> inline constexpr bool kAlwaysFalseOneSweepPartition = false;

template <class BlockHistogramPolicy, class LocalRankPolicy>
struct onesweep_block_digits_impl {
  static_assert(kAlwaysFalseOneSweepPartition<BlockHistogramPolicy>,
                "onesweep block digit policy combination is not implemented");
};

template <class LocalRankPolicy>
struct onesweep_block_digits_impl<WarpBallotBlockHistogram, LocalRankPolicy> {
  template <int BlockSize, int RadixBits, int ItemsPerThread>
  __device__ static void compute(
      std::uint32_t (&ranks)[ItemsPerThread], std::uint32_t *block_histogram,
      std::uint32_t (&digits)[ItemsPerThread],
      bool (&valid_items)[ItemsPerThread], std::uint32_t *warp_counts,
      std::uint32_t *warp_bucket_scratch) {
    using local_rank = histogram_local_rank_impl<LocalRankPolicy>;
    local_rank::template compute<BlockSize, RadixBits, ItemsPerThread>(
        ranks, block_histogram, digits, valid_items, warp_counts,
        warp_bucket_scratch);
  }
};

template <int BlockSize, int RadixBits, int ItemsPerThread,
          class BlockHistogramPolicy,
          class LocalRankPolicy, class Key>
__global__ void onesweep_partition_keys_kernel(
    Key *output, const Key *input, const std::uint32_t *global_offsets,
    onesweep_lookback_record *lookback_records,
    std::uint32_t *tile_counter, std::uint32_t count, int shift) {
  using block_digits =
      onesweep_block_digits_impl<BlockHistogramPolicy, LocalRankPolicy>;

  constexpr std::uint32_t kNumBuckets =
      static_cast<std::uint32_t>(1u << RadixBits);
  constexpr int kWarpSize = 32;
  constexpr int kWarpCount = BlockSize / kWarpSize;

  __shared__ std::uint32_t block_histogram[kNumBuckets];
  __shared__ std::uint32_t block_prefix[kNumBuckets];
  __shared__ std::uint32_t
      warp_counts[static_cast<std::size_t>(kWarpCount) * kNumBuckets];
  __shared__ std::uint32_t
      warp_bucket_scratch[static_cast<std::size_t>(kWarpCount) * kNumBuckets];

  __shared__ std::uint32_t tile_index;
  if (threadIdx.x == 0) {
    tile_index = atomicAdd(tile_counter, 1u);
  }
  __syncthreads();

  Key keys[ItemsPerThread];
  std::uint32_t digits[ItemsPerThread];
  std::uint32_t ranks[ItemsPerThread];
  bool valid_items[ItemsPerThread];
  const std::uint32_t tile_base =
      tile_index * static_cast<std::uint32_t>(BlockSize * ItemsPerThread);
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

  block_digits::template compute<BlockSize, RadixBits, ItemsPerThread>(
      ranks, block_histogram, digits, valid_items, warp_counts,
      warp_bucket_scratch);

  const std::uint32_t num_blocks = gridDim.x;
  for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
       bucket += blockDim.x) {
    auto *bucket_records =
        lookback_records + static_cast<std::size_t>(bucket) * num_blocks;
    onesweep_publish_aggregate(bucket_records, tile_index,
                               block_histogram[bucket]);
  }
  __syncthreads();

  for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
       bucket += blockDim.x) {
    auto *bucket_records =
        lookback_records + static_cast<std::size_t>(bucket) * num_blocks;
    const std::uint32_t prefix =
        tile_index == 0 ? 0u : onesweep_prefix(bucket_records, tile_index);
    block_prefix[bucket] = prefix;
    onesweep_publish_prefix(bucket_records, tile_index,
                            prefix + block_histogram[bucket]);
  }
  __syncthreads();

  for (int item = 0; item < ItemsPerThread; ++item) {
    if (valid_items[item]) {
      const std::uint32_t digit = digits[item];
      const std::uint32_t output_index =
          global_offsets[digit] + block_prefix[digit] + ranks[item];
      output[output_index] = keys[item];
    }
  }
}

template <int BlockSize, int RadixBits, int ItemsPerThread,
          class BlockHistogramPolicy,
          class LocalRankPolicy, class Key, class Value>
__global__ void onesweep_partition_pairs_kernel(
    Key *output_keys, Value *output_values, const Key *input_keys,
    const Value *input_values, const std::uint32_t *global_offsets,
    onesweep_lookback_record *lookback_records,
    std::uint32_t *tile_counter, std::uint32_t count, int shift) {
  using block_digits =
      onesweep_block_digits_impl<BlockHistogramPolicy, LocalRankPolicy>;

  constexpr std::uint32_t kNumBuckets =
      static_cast<std::uint32_t>(1u << RadixBits);
  constexpr int kWarpSize = 32;
  constexpr int kWarpCount = BlockSize / kWarpSize;

  __shared__ std::uint32_t block_histogram[kNumBuckets];
  __shared__ std::uint32_t block_prefix[kNumBuckets];
  __shared__ std::uint32_t
      warp_counts[static_cast<std::size_t>(kWarpCount) * kNumBuckets];
  __shared__ std::uint32_t
      warp_bucket_scratch[static_cast<std::size_t>(kWarpCount) * kNumBuckets];

  __shared__ std::uint32_t tile_index;
  if (threadIdx.x == 0) {
    tile_index = atomicAdd(tile_counter, 1u);
  }
  __syncthreads();

  Key keys[ItemsPerThread];
  Value values[ItemsPerThread];
  std::uint32_t digits[ItemsPerThread];
  std::uint32_t ranks[ItemsPerThread];
  bool valid_items[ItemsPerThread];
  const std::uint32_t tile_base =
      tile_index * static_cast<std::uint32_t>(BlockSize * ItemsPerThread);
  for (int item = 0; item < ItemsPerThread; ++item) {
    const std::uint32_t global_index =
        warp_major_tile_index<BlockSize, ItemsPerThread>(tile_base, item);
    const bool valid = global_index < count;
    const Key key = valid ? input_keys[global_index] : Key{};
    keys[item] = key;
    values[item] = valid ? input_values[global_index] : Value{};
    digits[item] = valid ? extract_digit<RadixBits>(key, shift) : 0u;
    ranks[item] = 0;
    valid_items[item] = valid;
  }

  block_digits::template compute<BlockSize, RadixBits, ItemsPerThread>(
      ranks, block_histogram, digits, valid_items, warp_counts,
      warp_bucket_scratch);

  const std::uint32_t num_blocks = gridDim.x;
  for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
       bucket += blockDim.x) {
    auto *bucket_records =
        lookback_records + static_cast<std::size_t>(bucket) * num_blocks;
    onesweep_publish_aggregate(bucket_records, tile_index,
                               block_histogram[bucket]);
  }
  __syncthreads();

  for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
       bucket += blockDim.x) {
    auto *bucket_records =
        lookback_records + static_cast<std::size_t>(bucket) * num_blocks;
    const std::uint32_t prefix =
        tile_index == 0 ? 0u : onesweep_prefix(bucket_records, tile_index);
    block_prefix[bucket] = prefix;
    onesweep_publish_prefix(bucket_records, tile_index,
                            prefix + block_histogram[bucket]);
  }
  __syncthreads();

  for (int item = 0; item < ItemsPerThread; ++item) {
    if (valid_items[item]) {
      const std::uint32_t digit = digits[item];
      const std::uint32_t output_index =
          global_offsets[digit] + block_prefix[digit] + ranks[item];
      output_keys[output_index] = keys[item];
      output_values[output_index] = values[item];
    }
  }
}

} // namespace algo::cuda::sort::detail
