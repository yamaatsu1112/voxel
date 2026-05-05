#pragma once

#include <algo/cuda/hash_table/acceleration_hash/common.cuh>

#include <cstdint>

namespace algo::cuda {

#if defined(__CUDACC__)
#define ALGO_CUDA_ACCEL_MAP_HD __host__ __device__
#else
#define ALGO_CUDA_ACCEL_MAP_HD
#endif

template <std::uint32_t ItemWords> struct AccelerationHashMapSlot32 {
  static_assert(ItemWords > 0,
                "AccelerationHashMapSlot32 requires item words");

  std::uint32_t item[ItemWords];
  std::uint32_t value;
};

template <std::uint32_t ItemWords>
struct alignas(128) AccelerationHashMapSlab32 {
  static_assert(ItemWords > 0,
                "AccelerationHashMapSlab32 requires item words");

  std::uint32_t accel_hash[kAccelerationHashSlotsPerSlab];
  std::uint32_t next;
  AccelerationHashMapSlot32<ItemWords> slots[kAccelerationHashSlotsPerSlab];

  ALGO_CUDA_ACCEL_MAP_HD static void initialize(AccelerationHashMapSlab32 *slab,
                                                std::uint32_t lane) {
    if (lane < kAccelerationHashSlotsPerSlab)
      slab->accel_hash[lane] = kAccelerationHashEmpty;
    if (lane == 31)
      slab->next = kSlabHashNullSlab;
  }
};

static_assert(alignof(AccelerationHashMapSlab32<1>) == 128);

template <std::uint32_t ItemWords, class Allocator>
struct DeviceAccelerationHashMap32 {
  using AllocatorPolicy = Allocator;
  using AllocatorDevice = typename Allocator::Device;
  using Slab = AccelerationHashMapSlab32<ItemWords>;

  Slab *slabs;
  std::uint32_t bucket_count;
  std::uint32_t slab_count;
  AllocatorDevice allocator;

#ifdef __CUDACC__
  __device__ void find(bool &is_active, const std::uint32_t *item,
                       std::uint32_t &ptr, std::uint32_t &value,
                       std::uint32_t &status) const;

  // Inserts without checking for an equal item. The caller must ensure that no
  // equal item is already present or concurrently inserted.
  __device__ void insert_unique_unchecked(bool &is_active,
                                          const std::uint32_t *item,
                                          std::uint32_t value,
                                          std::uint32_t &ptr,
                                          std::uint32_t &status);

  __device__ bool read_item(std::uint32_t ptr,
                            std::uint32_t (&out)[ItemWords],
                            std::uint32_t &value) const;
#endif
};

} // namespace algo::cuda

#undef ALGO_CUDA_ACCEL_MAP_HD
