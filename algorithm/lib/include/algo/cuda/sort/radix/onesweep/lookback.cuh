#pragma once

#include <algo/cuda/sort/radix/onesweep/layout.cuh>

namespace algo::cuda::sort::detail {

inline constexpr unsigned int kOneSweepLookbackEmpty = 0;
inline constexpr unsigned int kOneSweepLookbackAggregate = 1;
inline constexpr unsigned int kOneSweepLookbackPrefix = 2;
inline constexpr std::uint32_t kOneSweepLookbackStateShift = 30;
inline constexpr std::uint32_t kOneSweepLookbackValueMask =
    (1u << kOneSweepLookbackStateShift) - 1u;

__host__ __device__ __forceinline__ std::uint32_t
onesweep_pack_record(std::uint32_t state, std::uint32_t value) {
    return (state << kOneSweepLookbackStateShift) |
           (value & kOneSweepLookbackValueMask);
}

__host__ __device__ __forceinline__ std::uint32_t
onesweep_record_state(std::uint32_t word) {
    return word >> kOneSweepLookbackStateShift;
}

__host__ __device__ __forceinline__ std::uint32_t
onesweep_record_value(std::uint32_t word) {
    return word & kOneSweepLookbackValueMask;
}

__device__ __forceinline__ std::uint32_t
onesweep_load_record(volatile onesweep_lookback_record* records,
                     std::uint32_t index) {
    auto* word = const_cast<std::uint32_t*>(&records[index].word);
    return atomicAdd(word, 0u);
}

__device__ __forceinline__ void onesweep_publish_aggregate(
    volatile onesweep_lookback_record* records, std::uint32_t index,
    std::uint32_t aggregate) {
    auto* word = const_cast<std::uint32_t*>(&records[index].word);
    atomicExch(
        word, onesweep_pack_record(kOneSweepLookbackAggregate, aggregate));
}

__device__ __forceinline__ void onesweep_publish_prefix(
    volatile onesweep_lookback_record* records, std::uint32_t index,
    std::uint32_t prefix) {
    auto* word = const_cast<std::uint32_t*>(&records[index].word);
    atomicExch(word,
               onesweep_pack_record(kOneSweepLookbackPrefix, prefix));
}

__device__ std::uint32_t
onesweep_prefix(volatile onesweep_lookback_record* records,
                 std::uint32_t block_index) {
    std::uint32_t prefix = 0;
    std::uint32_t cursor = block_index;

    while (cursor > 0) {
        --cursor;

        std::uint32_t word = 0;
        std::uint32_t state = kOneSweepLookbackEmpty;
        while (state == kOneSweepLookbackEmpty) {
            word = onesweep_load_record(records, cursor);
            state = onesweep_record_state(word);
        }

        const std::uint32_t value = onesweep_record_value(word);
        if (state == kOneSweepLookbackPrefix) {
            prefix += value;
            break;
        }

        prefix += value;
    }

    return prefix;
}

} // namespace algo::cuda::sort::detail
