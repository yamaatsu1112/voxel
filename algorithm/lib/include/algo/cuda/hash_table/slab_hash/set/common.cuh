#pragma once

#include <algo/cuda/hash_table/slab_hash/common.cuh>

namespace algo::cuda {

#if defined(__CUDACC__)
#define ALGO_CUDA_HD __host__ __device__
#else
#define ALGO_CUDA_HD
#endif

inline constexpr std::uint32_t kSlabHashSetKeysPerSlab = 31;
inline constexpr std::uint32_t kSlabHashSetWordsPerSlab =
    kSlabHashSetKeysPerSlab + 1u;

struct alignas(128) SlabHashSetU32Slab {
  std::uint32_t keys[kSlabHashSetKeysPerSlab];
  std::uint32_t next;

  ALGO_CUDA_HD static void initialize(SlabHashSetU32Slab *slab,
                                      std::uint32_t lane) {
    if (lane < kSlabHashSetKeysPerSlab) {
      slab->keys[lane] = kSlabHashEmptyKey;
    }
    if (lane == 31)
      slab->next = kSlabHashNullSlab;
  }
};

static_assert(sizeof(SlabHashSetU32Slab) == 128);

template <class Allocator> struct DeviceSlabHashSetU32 {
  using AllocatorPolicy = Allocator;
  using AllocatorDevice = typename Allocator::Device;

  SlabHashSetU32Slab *slabs;
  std::uint32_t bucket_count;
  std::uint32_t slab_capacity;
  AllocatorDevice allocator;

#ifdef __CUDACC__
  __device__ void insert(bool &is_active, std::uint32_t &key,
                         std::uint32_t &status);

  __device__ void contains(bool &is_active, std::uint32_t &key,
                           std::uint32_t &contains,
                           std::uint32_t &status) const;

  __device__ void erase(bool &is_active, std::uint32_t &key,
                        std::uint32_t &status);
#endif
};

} // namespace algo::cuda

#undef ALGO_CUDA_HD
