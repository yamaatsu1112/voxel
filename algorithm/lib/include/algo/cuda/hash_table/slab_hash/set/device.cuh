#pragma once

#include <algo/cuda/hash_table/bump_allocator.cuh>
#include <algo/cuda/hash_table/slab_hash/set/common.cuh>

#ifdef __CUDACC__

#include <cuda_runtime.h>

namespace algo::cuda {

namespace detail::slab_hash_set {

inline constexpr std::uint32_t kFullWarpMask = 0xffffffffu;
inline constexpr std::uint32_t kKeyLaneMask = 0x7fffffffu;

__device__ inline std::uint32_t lane_id() {
  return static_cast<std::uint32_t>(threadIdx.x) & (kSlabHashWarpSize - 1u);
}

__device__ inline bool is_key_lane(std::uint32_t lane) {
  return lane < kSlabHashSetKeysPerSlab;
}

__device__ inline std::uint32_t atomic_load_u32(std::uint32_t *ptr) {
  return atomicAdd(ptr, 0u);
}

} // namespace detail::slab_hash_set

template <class Table>
__device__ inline void
slab_hash_set_contains(Table table, bool &is_active, std::uint32_t &key,
                       std::uint32_t &contains, std::uint32_t &status) {
  const std::uint32_t lane = detail::slab_hash_set::lane_id();
  if (is_active &&
      (table.bucket_count == 0 || slab_hash_is_reserved_key(key))) {
    contains = 0;
    status = kSlabHashStatusInvalidKey;
    is_active = false;
  }

  const std::uint32_t max_chain_length = table.slab_capacity + 1u;
  std::uint32_t work_queue =
      __ballot_sync(detail::slab_hash_set::kFullWarpMask, is_active);

  while (work_queue != 0) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(work_queue) - 1);
    const std::uint32_t src_key =
        __shfl_sync(detail::slab_hash_set::kFullWarpMask, key, src_lane);
    const std::uint32_t bucket_index =
        slab_hash_bucket(src_key, table.bucket_count);
    SlabHashSetU32Slab *slab =
        &table.slabs[slab_hash_bucket_storage_index(bucket_index)];
    bool done = false;

    for (std::uint32_t visited = 0; !done && visited < max_chain_length;
         ++visited) {
      const bool key_lane = detail::slab_hash_set::is_key_lane(lane);
      const std::uint32_t slot_key =
          key_lane ? detail::slab_hash_set::atomic_load_u32(&slab->keys[lane])
                   : 0u;
      const std::uint32_t match_mask =
          __ballot_sync(detail::slab_hash_set::kFullWarpMask,
                        key_lane && slot_key == src_key) &
          detail::slab_hash_set::kKeyLaneMask;

      if (match_mask != 0) {
        if (lane == src_lane) {
          contains = 1;
          status = kSlabHashStatusFound;
          is_active = false;
        }
        done = true;
        break;
      }

      std::uint32_t next =
          (lane == 31) ? detail::slab_hash_set::atomic_load_u32(&slab->next)
                       : kSlabHashNullSlab;
      next = __shfl_sync(detail::slab_hash_set::kFullWarpMask, next, 31);
      if (next == kSlabHashNullSlab) {
        if (lane == src_lane) {
          contains = 0;
          status = kSlabHashStatusNotFound;
          is_active = false;
        }
        done = true;
        break;
      }
      slab = &table.slabs[next];
    }

    if (!done && lane == src_lane) {
      contains = 0;
      status = kSlabHashStatusOverflow;
      is_active = false;
    }
    work_queue = __ballot_sync(detail::slab_hash_set::kFullWarpMask, is_active);
  }
}

template <class Table>
__device__ inline void slab_hash_set_insert(Table table, bool &is_active,
                                            std::uint32_t &key,
                                            std::uint32_t &status) {
  using Allocator = typename Table::AllocatorPolicy;

  const std::uint32_t lane = detail::slab_hash_set::lane_id();
  if (is_active &&
      (table.bucket_count == 0 || slab_hash_is_reserved_key(key))) {
    status = kSlabHashStatusInvalidKey;
    is_active = false;
  }

  const std::uint32_t max_attempts =
      (table.slab_capacity + 1u) * kSlabHashMaxContentionRetries;
  std::uint32_t work_queue =
      __ballot_sync(detail::slab_hash_set::kFullWarpMask, is_active);

  while (work_queue != 0) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(work_queue) - 1);
    const std::uint32_t src_key =
        __shfl_sync(detail::slab_hash_set::kFullWarpMask, key, src_lane);
    const std::uint32_t bucket_index =
        slab_hash_bucket(src_key, table.bucket_count);
    SlabHashSetU32Slab *slab =
        &table.slabs[slab_hash_bucket_storage_index(bucket_index)];
    bool done = false;

    for (std::uint32_t attempts = 0; !done && attempts < max_attempts;
         ++attempts) {
      const bool key_lane = detail::slab_hash_set::is_key_lane(lane);
      const std::uint32_t slot_key =
          key_lane ? detail::slab_hash_set::atomic_load_u32(&slab->keys[lane])
                   : 0u;
      const std::uint32_t match_mask =
          __ballot_sync(detail::slab_hash_set::kFullWarpMask,
                        key_lane && slot_key == src_key) &
          detail::slab_hash_set::kKeyLaneMask;
      if (match_mask != 0) {
        if (lane == src_lane) {
          status = kSlabHashStatusAlreadyPresent;
          is_active = false;
        }
        done = true;
        break;
      }

      const bool candidate = key_lane && (slot_key == kSlabHashEmptyKey ||
                                          slot_key == kSlabHashDeletedKey);
      const std::uint32_t candidate_mask =
          __ballot_sync(detail::slab_hash_set::kFullWarpMask, candidate) &
          detail::slab_hash_set::kKeyLaneMask;
      if (candidate_mask != 0) {
        const std::uint32_t dest_lane =
            static_cast<std::uint32_t>(__ffs(candidate_mask) - 1);
        const std::uint32_t expected_key = __shfl_sync(
            detail::slab_hash_set::kFullWarpMask, slot_key, dest_lane);
        std::uint32_t old_key = kSlabHashEmptyKey;
        if (lane == dest_lane) {
          old_key = atomicCAS(&slab->keys[dest_lane], expected_key, src_key);
        }
        old_key = __shfl_sync(detail::slab_hash_set::kFullWarpMask, old_key,
                              dest_lane);
        if (old_key == expected_key) {
          if (lane == src_lane) {
            status = kSlabHashStatusInserted;
            is_active = false;
          }
          done = true;
          break;
        }
        continue;
      }

      std::uint32_t next =
          (lane == 31) ? detail::slab_hash_set::atomic_load_u32(&slab->next)
                       : kSlabHashNullSlab;
      next = __shfl_sync(detail::slab_hash_set::kFullWarpMask, next, 31);
      if (next != kSlabHashNullSlab) {
        slab = &table.slabs[next];
        continue;
      }

      const std::uint32_t dynamic_ordinal =
          Allocator::allocate(table.allocator, lane);
      if (dynamic_ordinal == kSlabHashNullSlab) {
        std::uint32_t observed =
            (lane == 31) ? detail::slab_hash_set::atomic_load_u32(&slab->next)
                         : kSlabHashNullSlab;
        observed =
            __shfl_sync(detail::slab_hash_set::kFullWarpMask, observed, 31);
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
      SlabHashSetU32Slab::initialize(&table.slabs[allocated], lane);
      __syncwarp(detail::slab_hash_set::kFullWarpMask);
      __threadfence();
      __syncwarp(detail::slab_hash_set::kFullWarpMask);

      std::uint32_t observed = kSlabHashNullSlab;
      if (lane == 31) {
        observed = atomicCAS(&slab->next, kSlabHashNullSlab, allocated);
      }
      observed =
          __shfl_sync(detail::slab_hash_set::kFullWarpMask, observed, 31);
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
    work_queue = __ballot_sync(detail::slab_hash_set::kFullWarpMask, is_active);
  }
}

template <class Table>
__device__ inline void slab_hash_set_erase(Table table, bool &is_active,
                                           std::uint32_t &key,
                                           std::uint32_t &status) {
  const std::uint32_t lane = detail::slab_hash_set::lane_id();
  if (is_active &&
      (table.bucket_count == 0 || slab_hash_is_reserved_key(key))) {
    status = kSlabHashStatusInvalidKey;
    is_active = false;
  }

  const std::uint32_t max_attempts =
      (table.slab_capacity + 1u) * kSlabHashMaxContentionRetries;
  std::uint32_t work_queue =
      __ballot_sync(detail::slab_hash_set::kFullWarpMask, is_active);

  while (work_queue != 0) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(work_queue) - 1);
    const std::uint32_t src_key =
        __shfl_sync(detail::slab_hash_set::kFullWarpMask, key, src_lane);
    const std::uint32_t bucket_index =
        slab_hash_bucket(src_key, table.bucket_count);
    SlabHashSetU32Slab *slab =
        &table.slabs[slab_hash_bucket_storage_index(bucket_index)];
    bool done = false;

    for (std::uint32_t attempts = 0; !done && attempts < max_attempts;
         ++attempts) {
      const bool key_lane = detail::slab_hash_set::is_key_lane(lane);
      const std::uint32_t slot_key =
          key_lane ? detail::slab_hash_set::atomic_load_u32(&slab->keys[lane])
                   : 0u;
      const std::uint32_t match_mask =
          __ballot_sync(detail::slab_hash_set::kFullWarpMask,
                        key_lane && slot_key == src_key) &
          detail::slab_hash_set::kKeyLaneMask;
      if (match_mask != 0) {
        const std::uint32_t dest_lane =
            static_cast<std::uint32_t>(__ffs(match_mask) - 1);
        std::uint32_t old_key = kSlabHashEmptyKey;
        if (lane == dest_lane) {
          old_key =
              atomicCAS(&slab->keys[dest_lane], src_key, kSlabHashDeletedKey);
        }
        old_key = __shfl_sync(detail::slab_hash_set::kFullWarpMask, old_key,
                              dest_lane);
        if (old_key == src_key) {
          if (lane == src_lane) {
            status = kSlabHashStatusErased;
            is_active = false;
          }
          done = true;
          break;
        }
        continue;
      }

      std::uint32_t next =
          (lane == 31) ? detail::slab_hash_set::atomic_load_u32(&slab->next)
                       : kSlabHashNullSlab;
      next = __shfl_sync(detail::slab_hash_set::kFullWarpMask, next, 31);
      if (next == kSlabHashNullSlab) {
        if (lane == src_lane) {
          status = kSlabHashStatusNotFound;
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
    work_queue = __ballot_sync(detail::slab_hash_set::kFullWarpMask, is_active);
  }
}

template <class Allocator>
__device__ inline void
DeviceSlabHashSetU32<Allocator>::insert(bool &is_active, std::uint32_t &key,
                                        std::uint32_t &status) {
  slab_hash_set_insert(*this, is_active, key, status);
}

template <class Allocator>
__device__ inline void DeviceSlabHashSetU32<Allocator>::contains(
    bool &is_active, std::uint32_t &key, std::uint32_t &contains,
    std::uint32_t &status) const {
  slab_hash_set_contains(*this, is_active, key, contains, status);
}

template <class Allocator>
__device__ inline void
DeviceSlabHashSetU32<Allocator>::erase(bool &is_active, std::uint32_t &key,
                                       std::uint32_t &status) {
  slab_hash_set_erase(*this, is_active, key, status);
}

} // namespace algo::cuda

#endif
