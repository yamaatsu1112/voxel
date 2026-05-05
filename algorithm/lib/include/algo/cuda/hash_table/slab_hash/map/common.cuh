#pragma once

#include <algo/cuda/hash_table/slab_hash/common.cuh>

namespace algo::cuda {

#if defined(__CUDACC__)
#define ALGO_CUDA_HD __host__ __device__
#else
#define ALGO_CUDA_HD
#endif

using SlabHashMapSlotWord = unsigned long long;

inline constexpr std::uint32_t kSlabHashMapSlotsPerSlab = 15;
inline constexpr std::uint32_t kSlabHashMapSlotWordsPerSlab =
    2u * kSlabHashMapSlotsPerSlab;
inline constexpr std::uint32_t kSlabHashMapWordsPerSlab =
    kSlabHashMapSlotWordsPerSlab + 2u;

ALGO_CUDA_HD constexpr SlabHashMapSlotWord
slab_hash_map_pack_slot(std::uint32_t key, std::uint32_t value) {
  return (static_cast<SlabHashMapSlotWord>(value) << 32u) |
         static_cast<SlabHashMapSlotWord>(key);
}

ALGO_CUDA_HD constexpr std::uint32_t
slab_hash_map_slot_key(SlabHashMapSlotWord slot) {
  return static_cast<std::uint32_t>(slot);
}

ALGO_CUDA_HD constexpr std::uint32_t
slab_hash_map_slot_value(SlabHashMapSlotWord slot) {
  return static_cast<std::uint32_t>(slot >> 32u);
}

inline constexpr SlabHashMapSlotWord kSlabHashMapEmptySlot =
    slab_hash_map_pack_slot(kSlabHashEmptyKey, 0u);
inline constexpr SlabHashMapSlotWord kSlabHashMapDeletedSlot =
    slab_hash_map_pack_slot(kSlabHashDeletedKey, 0u);

struct alignas(8) SlabHashMapU32Slot {
  std::uint32_t key;
  std::uint32_t value;
};

static_assert(sizeof(SlabHashMapU32Slot) == sizeof(SlabHashMapSlotWord));
static_assert(alignof(SlabHashMapU32Slot) == alignof(SlabHashMapSlotWord));

struct alignas(128) SlabHashMapU32Slab {
  SlabHashMapU32Slot slots[kSlabHashMapSlotsPerSlab];
  std::uint32_t reserved;
  std::uint32_t next;

  ALGO_CUDA_HD static void initialize(SlabHashMapU32Slab *slab,
                                      std::uint32_t lane) {
    if (lane < kSlabHashMapSlotWordsPerSlab) {
      SlabHashMapU32Slot &slot = slab->slots[lane >> 1u];
      if ((lane & 1u) == 0) {
        slot.key = kSlabHashEmptyKey;
      } else {
        slot.value = 0;
      }
    }
    if (lane == 30)
      slab->reserved = 0;
    if (lane == 31)
      slab->next = kSlabHashNullSlab;
  }
};

static_assert(sizeof(SlabHashMapU32Slab) == 128);

template <class Allocator> struct DeviceSlabHashMapU32 {
  using AllocatorPolicy = Allocator;
  using AllocatorDevice = typename Allocator::Device;

  SlabHashMapU32Slab *slabs;
  std::uint32_t bucket_count;
  std::uint32_t slab_capacity;
  AllocatorDevice allocator;

#ifdef __CUDACC__
  __device__ void insert_or_replace(bool &is_active, std::uint32_t &key,
                                    std::uint32_t &value,
                                    std::uint32_t &status);

  __device__ void find(bool &is_active, std::uint32_t &key,
                       std::uint32_t &value, std::uint32_t &status) const;

  __device__ void erase_all(bool &is_active, std::uint32_t &key,
                            std::uint32_t &status);
#endif
};

} // namespace algo::cuda

#undef ALGO_CUDA_HD
