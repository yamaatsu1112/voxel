#pragma once

#include <algo/cuda/hash_table/acceleration_hash/common.cuh>

#include <cstdint>

namespace algo::cuda {

#if defined(__CUDACC__)
#define ALGO_CUDA_ACCEL_SET_HD __host__ __device__
#else
#define ALGO_CUDA_ACCEL_SET_HD
#endif

template <std::uint32_t ItemWords> struct alignas(128) AccelerationHashSlab32 {
  static_assert(ItemWords > 0, "AccelerationHashSlab32 requires item words");

  std::uint32_t accel_hash[kAccelerationHashSlotsPerSlab];
  std::uint32_t next;
  std::uint32_t items[kAccelerationHashSlotsPerSlab][ItemWords];

  ALGO_CUDA_ACCEL_SET_HD static void initialize(AccelerationHashSlab32 *slab,
                                                std::uint32_t lane) {
    if (lane < kAccelerationHashSlotsPerSlab)
      slab->accel_hash[lane] = kAccelerationHashEmpty;
    if (lane == 31)
      slab->next = kSlabHashNullSlab;
  }
};

static_assert(alignof(AccelerationHashSlab32<1>) == 128);

template <std::uint32_t ItemWords, class Allocator>
struct DeviceAccelerationHashSet32 {
  using AllocatorPolicy = Allocator;
  using AllocatorDevice = typename Allocator::Device;
  using Slab = AccelerationHashSlab32<ItemWords>;

  Slab *slabs;
  std::uint32_t bucket_count;
  std::uint32_t slab_count;
  AllocatorDevice allocator;

#ifdef __CUDACC__
  // Looks up currently published table contents. This is not an
  // insert-if-absent primitive and does not guarantee uniqueness against
  // concurrent unchecked inserts.
  __device__ void find(bool &is_active, const std::uint32_t *item,
                       std::uint32_t &ptr, std::uint32_t &status) const;

  // Inserts without checking for an equal item. The caller must ensure that no
  // equal item is already present or concurrently inserted.
  __device__ void insert_unique_unchecked(bool &is_active,
                                          const std::uint32_t *item,
                                          std::uint32_t &ptr,
                                          std::uint32_t &status);

  __device__ bool read_item(std::uint32_t ptr,
                            std::uint32_t (&out)[ItemWords]) const;
#endif
};

} // namespace algo::cuda

#undef ALGO_CUDA_ACCEL_SET_HD
