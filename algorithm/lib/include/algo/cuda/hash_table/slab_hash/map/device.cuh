#pragma once

#include <algo/cuda/hash_table/bump_allocator.cuh>
#include <algo/cuda/hash_table/slab_hash/map/common.cuh>

#ifdef __CUDACC__

#include <cuda_runtime.h>

namespace algo::cuda {

namespace detail::slab_hash_map {

inline constexpr std::uint32_t kFullWarpMask = 0xffffffffu;
inline constexpr std::uint32_t kSlotLaneMask = 0x15555555u;

__device__ inline std::uint32_t lane_id() {
  return static_cast<std::uint32_t>(threadIdx.x) & (kSlabHashWarpSize - 1u);
}

__device__ inline bool is_slot_owner_lane(std::uint32_t lane) {
  return ((kSlotLaneMask >> lane) & 1u) != 0;
}

__device__ inline bool is_slot_word_lane(std::uint32_t lane) {
  return lane < kSlabHashMapSlotWordsPerSlab;
}

__device__ inline std::uint32_t slot_index_from_lane(std::uint32_t lane) {
  return lane >> 1u;
}

__device__ inline std::uint32_t atomic_load_u32(std::uint32_t *ptr) {
  return atomicAdd(ptr, 0u);
}

__device__ inline std::uint32_t atomic_load_slot_word(SlabHashMapU32Slot *slot,
                                                      std::uint32_t lane) {
  return (lane & 1u) == 0 ? atomic_load_u32(&slot->key)
                          : atomic_load_u32(&slot->value);
}

__device__ inline SlabHashMapSlotWord *slot_word_ptr(SlabHashMapU32Slot *slot) {
  return reinterpret_cast<SlabHashMapSlotWord *>(slot);
}

} // namespace detail::slab_hash_map

template <class Table>
__device__ inline void
slab_hash_map_find(Table table, bool &is_active, std::uint32_t &key,
                   std::uint32_t &value, std::uint32_t &status) {
  const std::uint32_t lane = detail::slab_hash_map::lane_id();
  if (is_active &&
      (table.bucket_count == 0 || slab_hash_is_reserved_key(key))) {
    value = kSlabHashNotFoundValue;
    status = kSlabHashStatusInvalidKey;
    is_active = false;
  }

  // Valid chains are acyclic and can contain at most the bucket slab plus
  // every extra slab, so this limit is not needed for normal operation. It
  // only prevents an infinite traversal if in-range next links are corrupted
  // into a cycle.
  const std::uint32_t max_chain_length = table.slab_capacity + 1u;
  std::uint32_t work_queue =
      __ballot_sync(detail::slab_hash_map::kFullWarpMask, is_active);

  while (work_queue != 0) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(work_queue) - 1);
    const std::uint32_t src_key =
        __shfl_sync(detail::slab_hash_map::kFullWarpMask, key, src_lane);
    const std::uint32_t bucket_index =
        slab_hash_bucket(src_key, table.bucket_count);
    SlabHashMapU32Slab *slab =
        &table.slabs[slab_hash_bucket_storage_index(bucket_index)];
    bool done = false;

    for (std::uint32_t visited = 0; !done && visited < max_chain_length;
         ++visited) {
      const bool slot_word_lane =
          detail::slab_hash_map::is_slot_word_lane(lane);
      const bool slot_owner_lane =
          detail::slab_hash_map::is_slot_owner_lane(lane);
      const std::uint32_t slot_word =
          slot_word_lane
              ? detail::slab_hash_map::atomic_load_slot_word(
                    &slab->slots[detail::slab_hash_map::slot_index_from_lane(
                        lane)],
                    lane)
              : 0u;
      const std::uint32_t match_mask =
          __ballot_sync(detail::slab_hash_map::kFullWarpMask,
                        slot_owner_lane && slot_word == src_key) &
          detail::slab_hash_map::kSlotLaneMask;

      if (match_mask != 0) {
        const std::uint32_t found_lane =
            static_cast<std::uint32_t>(__ffs(match_mask) - 1);
        const std::uint32_t found_value = __shfl_sync(
            detail::slab_hash_map::kFullWarpMask, slot_word, found_lane + 1u);
        if (lane == src_lane) {
          value = found_value;
          status = kSlabHashStatusFound;
          is_active = false;
        }
        done = true;
        break;
      }

      std::uint32_t next =
          (lane == 31) ? detail::slab_hash_map::atomic_load_u32(&slab->next)
                       : kSlabHashNullSlab;
      next = __shfl_sync(detail::slab_hash_map::kFullWarpMask, next, 31);
      if (next == kSlabHashNullSlab) {
        if (lane == src_lane) {
          value = kSlabHashNotFoundValue;
          status = kSlabHashStatusNotFound;
          is_active = false;
        }
        done = true;
        break;
      }
      slab = &table.slabs[next];
    }

    if (!done && lane == src_lane) {
      value = kSlabHashNotFoundValue;
      status = kSlabHashStatusOverflow;
      is_active = false;
    }
    work_queue = __ballot_sync(detail::slab_hash_map::kFullWarpMask, is_active);
  }
}

template <class Table>
__device__ inline void
slab_hash_map_insert_or_replace(Table table, bool &is_active,
                                std::uint32_t &key, std::uint32_t &value,
                                std::uint32_t &status) {
  using Allocator = typename Table::AllocatorPolicy;

  const std::uint32_t lane = detail::slab_hash_map::lane_id();
  if (is_active &&
      (table.bucket_count == 0 || slab_hash_is_reserved_key(key))) {
    status = kSlabHashStatusInvalidKey;
    is_active = false;
  }

  const std::uint32_t max_attempts =
      (table.slab_capacity + 1u) * kSlabHashMaxContentionRetries;
  std::uint32_t work_queue =
      __ballot_sync(detail::slab_hash_map::kFullWarpMask, is_active);

  while (work_queue != 0) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(work_queue) - 1);
    const std::uint32_t src_key =
        __shfl_sync(detail::slab_hash_map::kFullWarpMask, key, src_lane);
    const std::uint32_t src_value =
        __shfl_sync(detail::slab_hash_map::kFullWarpMask, value, src_lane);
    const std::uint32_t bucket_index =
        slab_hash_bucket(src_key, table.bucket_count);
    SlabHashMapU32Slab *slab =
        &table.slabs[slab_hash_bucket_storage_index(bucket_index)];
    bool done = false;

    for (std::uint32_t attempts = 0; !done && attempts < max_attempts;
         ++attempts) {
      const bool slot_word_lane =
          detail::slab_hash_map::is_slot_word_lane(lane);
      const bool slot_owner_lane =
          detail::slab_hash_map::is_slot_owner_lane(lane);
      const std::uint32_t slot_word =
          slot_word_lane
              ? detail::slab_hash_map::atomic_load_slot_word(
                    &slab->slots[detail::slab_hash_map::slot_index_from_lane(
                        lane)],
                    lane)
              : 0u;
      const bool candidate =
          slot_owner_lane &&
          (slot_word == src_key || slot_word == kSlabHashEmptyKey ||
           slot_word == kSlabHashDeletedKey);
      const std::uint32_t candidate_mask =
          __ballot_sync(detail::slab_hash_map::kFullWarpMask, candidate) &
          detail::slab_hash_map::kSlotLaneMask;
      if (candidate_mask != 0) {
        const std::uint32_t dest_lane =
            static_cast<std::uint32_t>(__ffs(candidate_mask) - 1);
        const std::uint32_t expected_key = __shfl_sync(
            detail::slab_hash_map::kFullWarpMask, slot_word, dest_lane);
        const std::uint32_t expected_value = __shfl_sync(
            detail::slab_hash_map::kFullWarpMask, slot_word, dest_lane + 1u);
        const SlabHashMapSlotWord expected_slot =
            slab_hash_map_pack_slot(expected_key, expected_value);
        const SlabHashMapSlotWord desired_slot =
            slab_hash_map_pack_slot(src_key, src_value);
        SlabHashMapSlotWord old_slot = 0ull;
        if (lane == dest_lane) {
          old_slot = atomicCAS(
              detail::slab_hash_map::slot_word_ptr(
                  &slab->slots[detail::slab_hash_map::slot_index_from_lane(
                      dest_lane)]),
              expected_slot, desired_slot);
        }
        old_slot = __shfl_sync(detail::slab_hash_map::kFullWarpMask, old_slot,
                               dest_lane);
        if (old_slot == expected_slot) {
          if (lane == src_lane) {
            status = expected_key == src_key ? kSlabHashStatusReplaced
                                             : kSlabHashStatusInserted;
            is_active = false;
          }
          done = true;
          break;
        }
        continue;
      }

      std::uint32_t next =
          (lane == 31) ? detail::slab_hash_map::atomic_load_u32(&slab->next)
                       : kSlabHashNullSlab;
      next = __shfl_sync(detail::slab_hash_map::kFullWarpMask, next, 31);
      if (next != kSlabHashNullSlab) {
        slab = &table.slabs[next];
        continue;
      }

      const std::uint32_t dynamic_ordinal =
          Allocator::allocate(table.allocator, lane);
      if (dynamic_ordinal == kSlabHashNullSlab) {
        std::uint32_t observed =
            (lane == 31) ? detail::slab_hash_map::atomic_load_u32(&slab->next)
                         : kSlabHashNullSlab;
        observed =
            __shfl_sync(detail::slab_hash_map::kFullWarpMask, observed, 31);
        if (observed != kSlabHashNullSlab) {
          slab = &table.slabs[observed];
          continue;
        }
        if (lane == src_lane) {
          status = kSlabHashStatusOverflow;
          is_active = false;
        }
        done = true;
        break;
      }

      const std::uint32_t allocated = table.bucket_count + dynamic_ordinal;
      SlabHashMapU32Slab::initialize(&table.slabs[allocated], lane);
      __syncwarp(detail::slab_hash_map::kFullWarpMask);
      __threadfence();
      __syncwarp(detail::slab_hash_map::kFullWarpMask);

      std::uint32_t observed = kSlabHashNullSlab;
      if (lane == 31) {
        observed = atomicCAS(&slab->next, kSlabHashNullSlab, allocated);
      }
      observed =
          __shfl_sync(detail::slab_hash_map::kFullWarpMask, observed, 31);
      if (observed == kSlabHashNullSlab) {
        slab = &table.slabs[allocated];
      } else {
        Allocator::deallocate(table.allocator, dynamic_ordinal, lane);
        slab = &table.slabs[observed];
      }
    }

    if (!done && lane == src_lane) {
      status = kSlabHashStatusOverflow;
      is_active = false;
    }
    work_queue = __ballot_sync(detail::slab_hash_map::kFullWarpMask, is_active);
  }
}

template <class Table>
__device__ inline void slab_hash_map_erase_all(Table table, bool &is_active,
                                               std::uint32_t &key,
                                               std::uint32_t &status) {
  const std::uint32_t lane = detail::slab_hash_map::lane_id();
  if (is_active &&
      (table.bucket_count == 0 || slab_hash_is_reserved_key(key))) {
    status = kSlabHashStatusInvalidKey;
    is_active = false;
  }

  const std::uint32_t max_attempts =
      (table.slab_capacity + 1u) * kSlabHashMaxContentionRetries;
  std::uint32_t work_queue =
      __ballot_sync(detail::slab_hash_map::kFullWarpMask, is_active);

  while (work_queue != 0) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(work_queue) - 1);
    const std::uint32_t src_key =
        __shfl_sync(detail::slab_hash_map::kFullWarpMask, key, src_lane);
    const std::uint32_t bucket_index =
        slab_hash_bucket(src_key, table.bucket_count);
    SlabHashMapU32Slab *slab =
        &table.slabs[slab_hash_bucket_storage_index(bucket_index)];
    bool erased_any = false;
    bool done = false;

    for (std::uint32_t attempts = 0; !done && attempts < max_attempts;
         ++attempts) {
      const bool slot_word_lane =
          detail::slab_hash_map::is_slot_word_lane(lane);
      const bool slot_owner_lane =
          detail::slab_hash_map::is_slot_owner_lane(lane);
      const std::uint32_t slot_word =
          slot_word_lane
              ? detail::slab_hash_map::atomic_load_slot_word(
                    &slab->slots[detail::slab_hash_map::slot_index_from_lane(
                        lane)],
                    lane)
              : 0u;
      const std::uint32_t match_mask =
          __ballot_sync(detail::slab_hash_map::kFullWarpMask,
                        slot_owner_lane && slot_word == src_key) &
          detail::slab_hash_map::kSlotLaneMask;
      if (match_mask != 0) {
        const std::uint32_t dest_lane =
            static_cast<std::uint32_t>(__ffs(match_mask) - 1);
        const std::uint32_t expected_value = __shfl_sync(
            detail::slab_hash_map::kFullWarpMask, slot_word, dest_lane + 1u);
        const SlabHashMapSlotWord expected_slot =
            slab_hash_map_pack_slot(src_key, expected_value);
        const SlabHashMapSlotWord deleted_slot =
            slab_hash_map_pack_slot(kSlabHashDeletedKey, expected_value);
        SlabHashMapSlotWord old_slot = 0ull;
        if (lane == dest_lane) {
          old_slot = atomicCAS(
              detail::slab_hash_map::slot_word_ptr(
                  &slab->slots[detail::slab_hash_map::slot_index_from_lane(
                      dest_lane)]),
              expected_slot, deleted_slot);
        }
        old_slot = __shfl_sync(detail::slab_hash_map::kFullWarpMask, old_slot,
                               dest_lane);
        if (old_slot == expected_slot) {
          if (lane == src_lane)
            erased_any = true;
        }
        continue;
      }

      std::uint32_t next =
          (lane == 31) ? detail::slab_hash_map::atomic_load_u32(&slab->next)
                       : kSlabHashNullSlab;
      next = __shfl_sync(detail::slab_hash_map::kFullWarpMask, next, 31);
      if (next == kSlabHashNullSlab) {
        if (lane == src_lane) {
          status = erased_any ? kSlabHashStatusErased : kSlabHashStatusNotFound;
          is_active = false;
        }
        done = true;
        break;
      }
      slab = &table.slabs[next];
    }

    if (!done && lane == src_lane) {
      status = kSlabHashStatusOverflow;
      is_active = false;
    }
    work_queue = __ballot_sync(detail::slab_hash_map::kFullWarpMask, is_active);
  }
}

template <class Allocator>
__device__ inline void DeviceSlabHashMapU32<Allocator>::insert_or_replace(
    bool &is_active, std::uint32_t &key, std::uint32_t &value,
    std::uint32_t &status) {
  slab_hash_map_insert_or_replace(*this, is_active, key, value, status);
}

template <class Allocator>
__device__ inline void
DeviceSlabHashMapU32<Allocator>::find(bool &is_active, std::uint32_t &key,
                                      std::uint32_t &value,
                                      std::uint32_t &status) const {
  slab_hash_map_find(*this, is_active, key, value, status);
}

template <class Allocator>
__device__ inline void
DeviceSlabHashMapU32<Allocator>::erase_all(bool &is_active, std::uint32_t &key,
                                           std::uint32_t &status) {
  slab_hash_map_erase_all(*this, is_active, key, status);
}

} // namespace algo::cuda

#endif
