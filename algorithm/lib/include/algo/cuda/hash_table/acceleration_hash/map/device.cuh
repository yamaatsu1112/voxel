#pragma once

#include <algo/cuda/hash_table/acceleration_hash/map/common.cuh>

#ifdef __CUDACC__

#include <cuda_runtime.h>

namespace algo::cuda {

namespace acceleration_hash::map::detail {

template <std::uint32_t ItemWords, class Slab>
__device__ inline bool slot_matches(Slab *slab, std::uint32_t slot,
                                    std::uint32_t selected_word,
                                    std::uint32_t lane) {
  bool word_equal = false;
  if (lane < ItemWords)
    word_equal = slab->slots[slot].item[lane] == selected_word;
  const std::uint32_t equal_mask = __ballot_sync(
      acceleration_hash::detail::kFullWarpMask, lane < ItemWords && word_equal);
  return (equal_mask &
          acceleration_hash::detail::item_equal_expected_mask<ItemWords>()) ==
         acceleration_hash::detail::item_equal_expected_mask<ItemWords>();
}

} // namespace acceleration_hash::map::detail

template <std::uint32_t ItemWords, class Table>
__device__ inline void acceleration_hash_map_find(Table table, bool &is_active,
                                                  const std::uint32_t *item,
                                                  std::uint32_t &ptr,
                                                  std::uint32_t &value,
                                                  std::uint32_t &status) {
  static_assert(ItemWords > 0 && ItemWords <= 31,
                "Acceleration hash map device API supports 1..31 words");

  const std::uint32_t lane = acceleration_hash::detail::lane_id();
  if (is_active && table.bucket_count == 0) {
    ptr = kAccelerationHashNullPtr;
    value = 0;
    status = kAccelerationHashStatusOverflow;
    is_active = false;
  }

  const std::uint32_t max_chain_length =
      acceleration_hash::detail::overflow_slab_capacity(table) + 1u;
  std::uint32_t work_queue =
      __ballot_sync(acceleration_hash::detail::kFullWarpMask, is_active);

  while (work_queue != 0) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(work_queue) - 1);
    std::uint32_t selected_word = 0;
    std::uint32_t primary_hash = 0;
    std::uint32_t accel_hash = 0;
    acceleration_hash::detail::select_item_for_work<ItemWords>(
        item, lane, src_lane, selected_word, primary_hash, accel_hash);
    const std::uint32_t bucket = primary_hash % table.bucket_count;
    typename Table::Slab *slab = &table.slabs[bucket];
    std::uint32_t current_slab_index = bucket;
    bool done = false;

    for (std::uint32_t visited = 0; !done && visited < max_chain_length;
         ++visited) {
      const bool slot_lane = lane < kAccelerationHashSlotsPerSlab;
      const std::uint32_t slot_accel_hash =
          slot_lane ? acceleration_hash::detail::atomic_load_u32(
                          &slab->accel_hash[lane])
                    : 0u;
      std::uint32_t candidate_mask =
          __ballot_sync(acceleration_hash::detail::kFullWarpMask,
                        slot_lane && slot_accel_hash == accel_hash) &
          acceleration_hash::detail::kSlotLaneMask;

      while (candidate_mask != 0) {
        const std::uint32_t slot =
            static_cast<std::uint32_t>(__ffs(candidate_mask) - 1);
        const bool match =
            acceleration_hash::map::detail::slot_matches<ItemWords>(
                slab, slot, selected_word, lane);
        if (match) {
          std::uint32_t matched_value = 0;
          if (lane == ItemWords)
            matched_value = slab->slots[slot].value;
          matched_value = __shfl_sync(
              acceleration_hash::detail::kFullWarpMask, matched_value,
              ItemWords);
          if (lane == src_lane) {
            ptr = acceleration_hash::detail::acceleration_hash_encode_ptr(
                current_slab_index, slot);
            value = matched_value;
            status = kAccelerationHashStatusFound;
            is_active = false;
          }
          done = true;
          break;
        }
        candidate_mask &= candidate_mask - 1u;
      }
      if (done)
        break;

      std::uint32_t next =
          (lane == 31) ? acceleration_hash::detail::atomic_load_u32(&slab->next)
                       : kSlabHashNullSlab;
      next = __shfl_sync(acceleration_hash::detail::kFullWarpMask, next, 31);
      if (next == kSlabHashNullSlab ||
          !acceleration_hash::detail::valid_overflow_slab_index(table, next)) {
        if (lane == src_lane) {
          ptr = kAccelerationHashNullPtr;
          value = 0;
          status = kAccelerationHashStatusNotFound;
          is_active = false;
        }
        done = true;
        break;
      }
      slab = &table.slabs[next];
      current_slab_index = next;
    }

    if (!done && lane == src_lane) {
      ptr = kAccelerationHashNullPtr;
      value = 0;
      status = kAccelerationHashStatusOverflow;
      is_active = false;
    }
    work_queue =
        __ballot_sync(acceleration_hash::detail::kFullWarpMask, is_active);
  }
}

template <std::uint32_t ItemWords, class Table>
__device__ inline void acceleration_hash_map_insert_unique_unchecked(
    Table table, bool &is_active, const std::uint32_t *item,
    std::uint32_t value, std::uint32_t &ptr, std::uint32_t &status) {
  static_assert(ItemWords > 0 && ItemWords <= 31,
                "Acceleration hash map device API supports 1..31 words");
  using Allocator = typename Table::AllocatorPolicy;

  const std::uint32_t lane = acceleration_hash::detail::lane_id();
  if (is_active && table.bucket_count == 0) {
    ptr = kAccelerationHashNullPtr;
    status = kAccelerationHashStatusOverflow;
    is_active = false;
  }

  const std::uint32_t overflow_capacity =
      acceleration_hash::detail::overflow_slab_capacity(table);
  const std::uint32_t max_attempts =
      (overflow_capacity + 1u) * kSlabHashMaxContentionRetries;
  std::uint32_t work_queue =
      __ballot_sync(acceleration_hash::detail::kFullWarpMask, is_active);

  while (work_queue != 0) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(work_queue) - 1);
    const std::uint32_t selected_value = __shfl_sync(
        acceleration_hash::detail::kFullWarpMask, value, src_lane);
    std::uint32_t selected_word = 0;
    std::uint32_t primary_hash = 0;
    std::uint32_t accel_hash = 0;
    acceleration_hash::detail::select_item_for_work<ItemWords>(
        item, lane, src_lane, selected_word, primary_hash, accel_hash);
    const std::uint32_t bucket = primary_hash % table.bucket_count;
    typename Table::Slab *slab = &table.slabs[bucket];
    std::uint32_t current_slab_index = bucket;
    bool done = false;

    for (std::uint32_t attempts = 0; !done && attempts < max_attempts;
         ++attempts) {
      const bool slot_lane = lane < kAccelerationHashSlotsPerSlab;
      const std::uint32_t slot_accel_hash =
          slot_lane ? acceleration_hash::detail::atomic_load_u32(
                          &slab->accel_hash[lane])
                    : kAccelerationHashWriting;
      const std::uint32_t empty_mask =
          __ballot_sync(acceleration_hash::detail::kFullWarpMask,
                        slot_lane &&
                            slot_accel_hash == kAccelerationHashEmpty) &
          acceleration_hash::detail::kSlotLaneMask;

      if (empty_mask != 0) {
        const std::uint32_t slot =
            static_cast<std::uint32_t>(__ffs(empty_mask) - 1);
        std::uint32_t old = kAccelerationHashWriting;
        if (lane == slot) {
          old = atomicCAS(&slab->accel_hash[slot], kAccelerationHashEmpty,
                          kAccelerationHashWriting);
        }
        old = __shfl_sync(acceleration_hash::detail::kFullWarpMask, old, slot);
        if (old == kAccelerationHashEmpty) {
          if (lane < ItemWords)
            slab->slots[slot].item[lane] = selected_word;
          if (lane == ItemWords)
            slab->slots[slot].value = selected_value;
          __syncwarp(acceleration_hash::detail::kFullWarpMask);
          __threadfence();
          if (lane == slot)
            slab->accel_hash[slot] = accel_hash;
          if (lane == src_lane) {
            ptr = acceleration_hash::detail::acceleration_hash_encode_ptr(
                current_slab_index, slot);
            status = kAccelerationHashStatusInserted;
            is_active = false;
          }
          done = true;
          break;
        }
        continue;
      }

      std::uint32_t next =
          (lane == 31) ? acceleration_hash::detail::atomic_load_u32(&slab->next)
                       : kSlabHashNullSlab;
      next = __shfl_sync(acceleration_hash::detail::kFullWarpMask, next, 31);
      if (next != kSlabHashNullSlab &&
          acceleration_hash::detail::valid_overflow_slab_index(table, next)) {
        slab = &table.slabs[next];
        current_slab_index = next;
        continue;
      }
      if (next != kSlabHashNullSlab) {
        if (lane == src_lane) {
          ptr = kAccelerationHashNullPtr;
          status = kAccelerationHashStatusOverflow;
          is_active = false;
        }
        done = true;
        break;
      }

      const std::uint32_t allocated = Allocator::allocate(
          table.allocator, table.slabs, table.slab_count, lane);
      if (allocated == kSlabHashNullSlab) {
        std::uint32_t observed =
            (lane == 31)
                ? acceleration_hash::detail::atomic_load_u32(&slab->next)
                : kSlabHashNullSlab;
        observed =
            __shfl_sync(acceleration_hash::detail::kFullWarpMask, observed, 31);
        if (observed != kSlabHashNullSlab &&
            acceleration_hash::detail::valid_overflow_slab_index(table,
                                                                 observed)) {
          slab = &table.slabs[observed];
          current_slab_index = observed;
          continue;
        }
        if (lane == src_lane) {
          ptr = kAccelerationHashNullPtr;
          status = kAccelerationHashStatusOverflow;
          is_active = false;
        }
        done = true;
        break;
      }

      std::uint32_t observed = kSlabHashNullSlab;
      if (lane == 31)
        observed = atomicCAS(&slab->next, kSlabHashNullSlab, allocated);
      observed =
          __shfl_sync(acceleration_hash::detail::kFullWarpMask, observed, 31);
      if (observed == kSlabHashNullSlab) {
        slab = &table.slabs[allocated];
        current_slab_index = allocated;
      } else if (acceleration_hash::detail::valid_overflow_slab_index(
                     table, observed)) {
        Allocator::deallocate(table.allocator, allocated, lane);
        slab = &table.slabs[observed];
        current_slab_index = observed;
      } else {
        Allocator::deallocate(table.allocator, allocated, lane);
        if (lane == src_lane) {
          ptr = kAccelerationHashNullPtr;
          status = kAccelerationHashStatusOverflow;
          is_active = false;
        }
        done = true;
        break;
      }
    }

    if (!done && lane == src_lane) {
      ptr = kAccelerationHashNullPtr;
      status = kAccelerationHashStatusOverflow;
      is_active = false;
    }
    work_queue =
        __ballot_sync(acceleration_hash::detail::kFullWarpMask, is_active);
  }
}

template <std::uint32_t ItemWords, class Table>
__device__ inline bool acceleration_hash_map_read_item(
    Table table, std::uint32_t ptr, std::uint32_t (&out)[ItemWords],
    std::uint32_t &value) {
  if (ptr == kAccelerationHashNullPtr)
    return false;
  const std::uint32_t slab_index =
      acceleration_hash::detail::acceleration_hash_ptr_slab_index(ptr);
  const std::uint32_t slot =
      acceleration_hash::detail::acceleration_hash_ptr_slot(ptr);
  if (slot >= kAccelerationHashSlotsPerSlab || slab_index >= table.slab_count)
    return false;

  typename Table::Slab *slab = &table.slabs[slab_index];
  for (std::uint32_t i = 0; i < ItemWords; ++i)
    out[i] = slab->slots[slot].item[i];
  value = slab->slots[slot].value;
  return true;
}

template <std::uint32_t ItemWords, class Allocator>
__device__ inline void DeviceAccelerationHashMap32<ItemWords, Allocator>::find(
    bool &is_active, const std::uint32_t *item,
    std::uint32_t &ptr, std::uint32_t &value, std::uint32_t &status) const {
  acceleration_hash_map_find<ItemWords>(*this, is_active, item, ptr, value,
                                         status);
}

template <std::uint32_t ItemWords, class Allocator>
__device__ inline void
DeviceAccelerationHashMap32<ItemWords, Allocator>::insert_unique_unchecked(
    bool &is_active, const std::uint32_t *item, std::uint32_t value,
    std::uint32_t &ptr, std::uint32_t &status) {
  acceleration_hash_map_insert_unique_unchecked<ItemWords>(
      *this, is_active, item, value, ptr, status);
}

template <std::uint32_t ItemWords, class Allocator>
__device__ inline bool
DeviceAccelerationHashMap32<ItemWords, Allocator>::read_item(
    std::uint32_t ptr, std::uint32_t (&out)[ItemWords],
    std::uint32_t &value) const {
  return acceleration_hash_map_read_item<ItemWords>(*this, ptr, out, value);
}

} // namespace algo::cuda

#endif
