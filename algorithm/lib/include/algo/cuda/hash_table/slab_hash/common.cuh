#pragma once

#include <cstdint>

namespace algo::cuda {

#if defined(__CUDACC__)
#define ALGO_CUDA_HD __host__ __device__
#else
#define ALGO_CUDA_HD
#endif

inline constexpr std::uint32_t kSlabHashEmptyKey = 0xffffffffu;
inline constexpr std::uint32_t kSlabHashDeletedKey = 0xfffffffeu;
inline constexpr std::uint32_t kSlabHashNullSlab = 0xffffffffu;
inline constexpr std::uint32_t kSlabHashNotFoundValue = 0xffffffffu;

inline constexpr std::uint32_t kSlabHashStatusNotFound = 0;
inline constexpr std::uint32_t kSlabHashStatusInserted = 1;
inline constexpr std::uint32_t kSlabHashStatusReplaced = 2;
inline constexpr std::uint32_t kSlabHashStatusErased = 3;
inline constexpr std::uint32_t kSlabHashStatusOverflow = 4;
inline constexpr std::uint32_t kSlabHashStatusInvalidKey = 5;
inline constexpr std::uint32_t kSlabHashStatusFound = 6;
inline constexpr std::uint32_t kSlabHashStatusAlreadyPresent = 7;

inline constexpr std::uint32_t kSlabHashWarpSize = 32;
inline constexpr std::uint32_t kSlabHashMaxContentionRetries = 65536;

ALGO_CUDA_HD constexpr bool slab_hash_is_reserved_key(std::uint32_t key) {
  return key == kSlabHashEmptyKey || key == kSlabHashDeletedKey;
}

ALGO_CUDA_HD constexpr std::uint32_t slab_hash_mix(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

ALGO_CUDA_HD constexpr std::uint32_t
slab_hash_bucket(std::uint32_t key, std::uint32_t bucket_count) {
  return slab_hash_mix(key) % bucket_count;
}

ALGO_CUDA_HD constexpr std::uint32_t
slab_hash_bucket_storage_index(std::uint32_t bucket_index) {
  return bucket_index;
}

ALGO_CUDA_HD constexpr std::uint32_t
slab_hash_allocated_slab_storage_begin(std::uint32_t bucket_count) {
  return bucket_count;
}

ALGO_CUDA_HD constexpr std::uint32_t
slab_hash_slab_storage_count(std::uint32_t bucket_count,
                             std::uint32_t slab_capacity) {
  return bucket_count + slab_capacity;
}

} // namespace algo::cuda

#undef ALGO_CUDA_HD
