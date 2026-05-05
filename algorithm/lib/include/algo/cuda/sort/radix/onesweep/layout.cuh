#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/common.cuh>

namespace algo::cuda::sort::detail {

struct onesweep_lookback_record {
  std::uint32_t word;
};

template <class BlockHistogramPolicy, class LocalRankPolicy,
          class GlobalOffsetsBlockHistogramPolicy, int ItemsPerThread,
          int BlockSize, int RadixBits, int KeyBits>
struct workspace_layout<
    RadixSort<OneSweepPass<BlockHistogramPolicy, LocalRankPolicy,
                           GlobalOffsetsBlockHistogramPolicy, ItemsPerThread>,
              BlockSize, RadixBits, KeyBits>> {
  static constexpr std::uint32_t kNumBuckets =
      static_cast<std::uint32_t>(1u << RadixBits);
  static constexpr std::uint32_t kPassCount =
      static_cast<std::uint32_t>(KeyBits / RadixBits);

  std::uint32_t *temp_keys = nullptr;
  std::uint32_t *global_offsets = nullptr;
  std::uint32_t *tile_counter = nullptr;
  onesweep_lookback_record *lookback_records = nullptr;
  std::uint32_t num_blocks = 0;

  static std::uint32_t block_count(std::uint32_t count) {
    return ::algo::ceil_div(
        count, static_cast<std::uint32_t>(BlockSize * ItemsPerThread));
  }

  static std::size_t required_workspace_size(std::uint32_t count) {
    if (count <= 1)
      return 0;

    const std::uint32_t blocks = block_count(count);
    std::size_t offset = 0;

    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * static_cast<std::size_t>(kPassCount) *
              kNumBuckets;

    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t);

    offset = align_up<onesweep_lookback_record>(offset);
    offset += sizeof(onesweep_lookback_record) *
              static_cast<std::size_t>(kNumBuckets) * blocks;

    return offset;
  }

  static workspace_layout create(void *workspace, std::uint32_t count) {
    workspace_layout layout{};
    layout.num_blocks = block_count(count);

    std::size_t offset = 0;

    offset = align_up<std::uint32_t>(offset);
    layout.temp_keys = pointer_at<std::uint32_t>(workspace, offset);
    offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

    offset = align_up<std::uint32_t>(offset);
    layout.global_offsets = pointer_at<std::uint32_t>(workspace, offset);
    offset += sizeof(std::uint32_t) * static_cast<std::size_t>(kPassCount) *
              kNumBuckets;

    offset = align_up<std::uint32_t>(offset);
    layout.tile_counter = pointer_at<std::uint32_t>(workspace, offset);
    offset += sizeof(std::uint32_t);

    offset = align_up<onesweep_lookback_record>(offset);
    layout.lookback_records =
        pointer_at<onesweep_lookback_record>(workspace, offset);

    return layout;
  }
};

template <class BlockHistogramPolicy, class LocalRankPolicy,
          class GlobalOffsetsBlockHistogramPolicy, int ItemsPerThread,
          int BlockSize, int RadixBits, int KeyBits, class Value>
struct pair_workspace_layout<
    RadixSort<OneSweepPass<BlockHistogramPolicy, LocalRankPolicy,
                           GlobalOffsetsBlockHistogramPolicy, ItemsPerThread>,
              BlockSize, RadixBits, KeyBits>,
    Value> {
  static constexpr std::uint32_t kNumBuckets =
      static_cast<std::uint32_t>(1u << RadixBits);
  static constexpr std::uint32_t kPassCount =
      static_cast<std::uint32_t>(KeyBits / RadixBits);

  std::uint32_t *temp_keys = nullptr;
  Value *temp_values = nullptr;
  std::uint32_t *global_offsets = nullptr;
  std::uint32_t *tile_counter = nullptr;
  onesweep_lookback_record *lookback_records = nullptr;
  std::uint32_t num_blocks = 0;

  static std::uint32_t block_count(std::uint32_t count) {
    return ::algo::ceil_div(
        count, static_cast<std::uint32_t>(BlockSize * ItemsPerThread));
  }

  static std::size_t required_workspace_size(std::uint32_t count) {
    if (count <= 1)
      return 0;

    const std::uint32_t blocks = block_count(count);
    std::size_t offset = 0;

    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

    offset = align_up<Value>(offset);
    offset += sizeof(Value) * static_cast<std::size_t>(count);

    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * static_cast<std::size_t>(kPassCount) *
              kNumBuckets;

    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t);

    offset = align_up<onesweep_lookback_record>(offset);
    offset += sizeof(onesweep_lookback_record) *
              static_cast<std::size_t>(kNumBuckets) * blocks;

    return offset;
  }

  static pair_workspace_layout create(void *workspace, std::uint32_t count) {
    pair_workspace_layout layout{};
    layout.num_blocks = block_count(count);

    std::size_t offset = 0;

    offset = align_up<std::uint32_t>(offset);
    layout.temp_keys = pointer_at<std::uint32_t>(workspace, offset);
    offset += sizeof(std::uint32_t) * static_cast<std::size_t>(count);

    offset = align_up<Value>(offset);
    layout.temp_values = pointer_at<Value>(workspace, offset);
    offset += sizeof(Value) * static_cast<std::size_t>(count);

    offset = align_up<std::uint32_t>(offset);
    layout.global_offsets = pointer_at<std::uint32_t>(workspace, offset);
    offset += sizeof(std::uint32_t) * static_cast<std::size_t>(kPassCount) *
              kNumBuckets;

    offset = align_up<std::uint32_t>(offset);
    layout.tile_counter = pointer_at<std::uint32_t>(workspace, offset);
    offset += sizeof(std::uint32_t);

    offset = align_up<onesweep_lookback_record>(offset);
    layout.lookback_records =
        pointer_at<onesweep_lookback_record>(workspace, offset);

    return layout;
  }
};

} // namespace algo::cuda::sort::detail
