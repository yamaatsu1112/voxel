#pragma once

#include <algo/cuda/hash_table/bump_allocator.cuh>
#include <algo/cuda/hash_table/slab_hash/common.cuh>

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace algo::cuda {

#if defined(__CUDACC__)
#define ALGO_CUDA_ACCEL_HD __host__ __device__
#else
#define ALGO_CUDA_ACCEL_HD
#endif

inline constexpr std::uint32_t kAccelerationHashSlotsPerSlab = 31;
inline constexpr std::uint32_t kAccelerationHashEmpty = 0;
inline constexpr std::uint32_t kAccelerationHashWriting = 1;
inline constexpr std::uint32_t kAccelerationHashNullPtr = 0;

inline constexpr std::uint32_t kAccelerationHashStatusNotFound =
    kSlabHashStatusNotFound;
inline constexpr std::uint32_t kAccelerationHashStatusInserted =
    kSlabHashStatusInserted;
inline constexpr std::uint32_t kAccelerationHashStatusOverflow =
    kSlabHashStatusOverflow;
inline constexpr std::uint32_t kAccelerationHashStatusFound =
    kSlabHashStatusFound;

namespace acceleration_hash::detail {

ALGO_CUDA_ACCEL_HD constexpr std::uint32_t
acceleration_hash_remap_code(std::uint32_t hash) {
  return hash < 2 ? hash + 2 : hash;
}

ALGO_CUDA_ACCEL_HD constexpr std::uint32_t
acceleration_hash_encode_ptr(std::uint32_t slab_index, std::uint32_t slot) {
  return ((slab_index << 5u) | slot) + 1u;
}

ALGO_CUDA_ACCEL_HD constexpr std::uint32_t
acceleration_hash_ptr_slab_index(std::uint32_t ptr) {
  return (ptr - 1u) >> 5u;
}

ALGO_CUDA_ACCEL_HD constexpr std::uint32_t
acceleration_hash_ptr_slot(std::uint32_t ptr) {
  return (ptr - 1u) & 31u;
}

ALGO_CUDA_ACCEL_HD inline std::uint32_t
acceleration_hash_mix_word(std::uint32_t hash, std::uint32_t word) {
  word *= 0xcc9e2d51u;
  word = (word << 15u) | (word >> 17u);
  word *= 0x1b873593u;

  hash ^= word;
  hash = (hash << 13u) | (hash >> 19u);
  return hash * 5u + 0xe6546b64u;
}

ALGO_CUDA_ACCEL_HD inline std::uint32_t
acceleration_hash_finish(std::uint32_t hash, std::uint32_t byte_count) {
  hash ^= byte_count;
  hash ^= hash >> 16u;
  hash *= 0x85ebca6bu;
  hash ^= hash >> 13u;
  hash *= 0xc2b2ae35u;
  hash ^= hash >> 16u;
  return hash;
}

template <std::uint32_t ItemWords>
ALGO_CUDA_ACCEL_HD inline std::uint32_t
acceleration_hash_words(const std::uint32_t (&words)[ItemWords],
                        std::uint32_t seed) {
  std::uint32_t hash = seed;
  for (std::uint32_t i = 0; i < ItemWords; ++i)
    hash = acceleration_hash_mix_word(hash, words[i]);
  return acceleration_hash_finish(hash, ItemWords * sizeof(std::uint32_t));
}

template <std::uint32_t ItemWords>
ALGO_CUDA_ACCEL_HD inline std::uint32_t
acceleration_hash_words(const std::uint32_t *words, std::uint32_t seed) {
  std::uint32_t hash = seed;
  for (std::uint32_t i = 0; i < ItemWords; ++i)
    hash = acceleration_hash_mix_word(hash, words[i]);
  return acceleration_hash_finish(hash, ItemWords * sizeof(std::uint32_t));
}

template <std::uint32_t ItemWords>
ALGO_CUDA_ACCEL_HD inline std::uint32_t
acceleration_hash_primary(const std::uint32_t (&words)[ItemWords]) {
  return acceleration_hash_words<ItemWords>(words, 0x31415927u);
}

template <std::uint32_t ItemWords>
ALGO_CUDA_ACCEL_HD inline std::uint32_t
acceleration_hash_primary(const std::uint32_t *words) {
  return acceleration_hash_words<ItemWords>(words, 0x31415927u);
}

template <std::uint32_t ItemWords>
ALGO_CUDA_ACCEL_HD inline std::uint32_t
acceleration_hash_accel_hash(const std::uint32_t (&words)[ItemWords]) {
  return acceleration_hash_remap_code(
      acceleration_hash_words<ItemWords>(words, 0x9e3779b9u));
}

template <std::uint32_t ItemWords>
ALGO_CUDA_ACCEL_HD inline std::uint32_t
acceleration_hash_accel_hash(const std::uint32_t *words) {
  return acceleration_hash_remap_code(
      acceleration_hash_words<ItemWords>(words, 0x9e3779b9u));
}

#ifdef __CUDACC__

inline constexpr std::uint32_t kFullWarpMask = 0xffffffffu;
inline constexpr std::uint32_t kSlotLaneMask = 0x7fffffffu;

__device__ inline std::uint32_t lane_id() {
  return static_cast<std::uint32_t>(threadIdx.x) & (kSlabHashWarpSize - 1u);
}

__device__ inline std::uint32_t atomic_load_u32(std::uint32_t *ptr) {
  return atomicAdd(ptr, 0u);
}

template <std::uint32_t ItemWords>
__device__ inline void select_item_for_work(const std::uint32_t *item,
                                            std::uint32_t lane,
                                            std::uint32_t src_lane,
                                            std::uint32_t &selected_word,
                                            std::uint32_t &primary_hash,
                                            std::uint32_t &accel_hash) {
  selected_word = 0;
  std::uint32_t lane_primary = 0x31415927u;
  std::uint32_t lane_accel_hash = 0x9e3779b9u;
#pragma unroll
  for (std::uint32_t i = 0; i < ItemWords; ++i) {
    std::uint32_t word = 0;
    if (lane == src_lane)
      word = item[i];
    word = __shfl_sync(kFullWarpMask, word, src_lane);
    lane_primary = acceleration_hash_mix_word(lane_primary, word);
    lane_accel_hash = acceleration_hash_mix_word(lane_accel_hash, word);
    if (lane == i)
      selected_word = word;
  }
  primary_hash =
      acceleration_hash_finish(lane_primary,
                               ItemWords * sizeof(std::uint32_t));
  accel_hash = acceleration_hash_remap_code(acceleration_hash_finish(
      lane_accel_hash, ItemWords * sizeof(std::uint32_t)));
}

template <class Table>
__device__ inline std::uint32_t overflow_slab_capacity(Table table) {
  return table.slab_count > table.bucket_count
             ? table.slab_count - table.bucket_count
             : 0u;
}

template <class Table>
__device__ inline bool valid_overflow_slab_index(Table table,
                                                 std::uint32_t slab_index) {
  return slab_index >= table.bucket_count && slab_index < table.slab_count;
}

template <std::uint32_t ItemWords>
__device__ inline std::uint32_t item_equal_expected_mask() {
  return ItemWords == 32 ? 0xffffffffu : ((1u << ItemWords) - 1u);
}

inline dim3 launch_grid(std::uint32_t count,
                        std::uint32_t block_threads = 256) {
  return dim3((count + block_threads - 1u) / block_threads);
}

#endif

} // namespace acceleration_hash::detail

#undef ALGO_CUDA_ACCEL_HD

} // namespace algo::cuda
