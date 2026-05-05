#pragma once

#include <algo/cuda/hash_table/bump_allocator.cuh>
#include <algo/cuda/hash_table/slab_hash/map/device.cuh>

#ifdef __CUDACC__

#include <cuda_runtime.h>

namespace algo::cuda {

template <class Allocator>
__global__ void
initialize_slab_hash_map_kernel(DeviceSlabHashMapU32<Allocator> table) {
  const std::uint32_t word_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t total_slabs =
      slab_hash_slab_storage_count(table.bucket_count, table.slab_capacity);
  Allocator::initialize(table.allocator);
  if (word_index >= total_slabs * kSlabHashMapWordsPerSlab)
    return;

  const std::uint32_t slab_index = word_index / kSlabHashMapWordsPerSlab;
  const std::uint32_t word = word_index % kSlabHashMapWordsPerSlab;
  SlabHashMapU32Slab::initialize(&table.slabs[slab_index], word);
}

template <class Allocator>
__global__ void slab_hash_map_insert_or_replace_kernel(
    DeviceSlabHashMapU32<Allocator> table, const std::uint32_t *keys,
    const std::uint32_t *values, std::uint32_t count, std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_active = index < count;
  std::uint32_t key = is_active ? keys[index] : 0;
  std::uint32_t value = is_active ? values[index] : 0;
  std::uint32_t status = kSlabHashStatusNotFound;

  table.insert_or_replace(is_active, key, value, status);
  if (index < count && statuses != nullptr)
    statuses[index] = status;
}

template <class Allocator>
__global__ void
slab_hash_map_find_kernel(DeviceSlabHashMapU32<Allocator> table,
                          const std::uint32_t *keys, std::uint32_t *values,
                          std::uint32_t count, std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_active = index < count;
  std::uint32_t key = is_active ? keys[index] : 0;
  std::uint32_t value = kSlabHashNotFoundValue;
  std::uint32_t status = kSlabHashStatusNotFound;

  table.find(is_active, key, value, status);
  if (index < count) {
    if (values != nullptr)
      values[index] = value;
    if (statuses != nullptr)
      statuses[index] = status;
  }
}

template <class Allocator>
__global__ void
slab_hash_map_erase_all_kernel(DeviceSlabHashMapU32<Allocator> table,
                               const std::uint32_t *keys, std::uint32_t count,
                               std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_active = index < count;
  std::uint32_t key = is_active ? keys[index] : 0;
  std::uint32_t status = kSlabHashStatusNotFound;

  table.erase_all(is_active, key, status);
  if (index < count && statuses != nullptr)
    statuses[index] = status;
}

inline std::uint32_t slab_hash_map_thread_count(std::uint32_t operation_count) {
  return operation_count;
}

inline dim3 slab_hash_map_grid(std::uint32_t operation_count,
                               std::uint32_t block_threads = 256) {
  const std::uint32_t threads = slab_hash_map_thread_count(operation_count);
  return dim3((threads + block_threads - 1u) / block_threads);
}

template <class Allocator>
inline cudaError_t
initialize_slab_hash_map(DeviceSlabHashMapU32<Allocator> table,
                         cudaStream_t stream = nullptr) {
  constexpr std::uint32_t kBlockThreads = 256;
  const std::uint32_t total_words =
      slab_hash_slab_storage_count(table.bucket_count, table.slab_capacity) *
      kSlabHashMapWordsPerSlab;
  const dim3 grid((total_words + kBlockThreads - 1u) / kBlockThreads);
  initialize_slab_hash_map_kernel<Allocator>
      <<<grid, kBlockThreads, 0, stream>>>(table);
  return cudaGetLastError();
}

template <class Allocator>
inline cudaError_t slab_hash_map_insert_or_replace_batch(
    DeviceSlabHashMapU32<Allocator> table, const std::uint32_t *keys,
    const std::uint32_t *values, std::uint32_t count,
    std::uint32_t *statuses = nullptr, cudaStream_t stream = nullptr) {
  if (count == 0)
    return cudaSuccess;
  constexpr std::uint32_t kBlockThreads = 256;
  slab_hash_map_insert_or_replace_kernel<Allocator>
      <<<slab_hash_map_grid(count, kBlockThreads), kBlockThreads, 0, stream>>>(
          table, keys, values, count, statuses);
  return cudaGetLastError();
}

template <class Allocator>
inline cudaError_t
slab_hash_map_find_batch(DeviceSlabHashMapU32<Allocator> table,
                         const std::uint32_t *keys, std::uint32_t *values,
                         std::uint32_t count, std::uint32_t *statuses = nullptr,
                         cudaStream_t stream = nullptr) {
  if (count == 0)
    return cudaSuccess;
  constexpr std::uint32_t kBlockThreads = 256;
  slab_hash_map_find_kernel<Allocator>
      <<<slab_hash_map_grid(count, kBlockThreads), kBlockThreads, 0, stream>>>(
          table, keys, values, count, statuses);
  return cudaGetLastError();
}

template <class Allocator>
inline cudaError_t
slab_hash_map_erase_all_batch(DeviceSlabHashMapU32<Allocator> table,
                              const std::uint32_t *keys, std::uint32_t count,
                              std::uint32_t *statuses = nullptr,
                              cudaStream_t stream = nullptr) {
  if (count == 0)
    return cudaSuccess;
  constexpr std::uint32_t kBlockThreads = 256;
  slab_hash_map_erase_all_kernel<Allocator>
      <<<slab_hash_map_grid(count, kBlockThreads), kBlockThreads, 0, stream>>>(
          table, keys, count, statuses);
  return cudaGetLastError();
}

} // namespace algo::cuda

#endif
