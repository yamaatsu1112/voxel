#pragma once

#include <algo/cuda/hash_table/bump_allocator.cuh>
#include <algo/cuda/hash_table/slab_hash/set/device.cuh>

#ifdef __CUDACC__

#include <cuda_runtime.h>

namespace algo::cuda {

template <class Allocator>
__global__ void
initialize_slab_hash_set_kernel(DeviceSlabHashSetU32<Allocator> table) {
  const std::uint32_t word_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t total_slabs =
      slab_hash_slab_storage_count(table.bucket_count, table.slab_capacity);
  Allocator::initialize(table.allocator);
  if (word_index >= total_slabs * kSlabHashSetWordsPerSlab)
    return;

  const std::uint32_t slab_index = word_index / kSlabHashSetWordsPerSlab;
  const std::uint32_t word = word_index % kSlabHashSetWordsPerSlab;
  SlabHashSetU32Slab::initialize(&table.slabs[slab_index], word);
}

template <class Allocator>
__global__ void
slab_hash_set_insert_kernel(DeviceSlabHashSetU32<Allocator> table,
                            const std::uint32_t *keys, std::uint32_t count,
                            std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_active = index < count;
  std::uint32_t key = is_active ? keys[index] : 0;
  std::uint32_t status = kSlabHashStatusNotFound;

  table.insert(is_active, key, status);
  if (index < count && statuses != nullptr)
    statuses[index] = status;
}

template <class Allocator>
__global__ void slab_hash_set_contains_kernel(
    DeviceSlabHashSetU32<Allocator> table, const std::uint32_t *keys,
    std::uint32_t *contains, std::uint32_t count, std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_active = index < count;
  std::uint32_t key = is_active ? keys[index] : 0;
  std::uint32_t contains_value = 0;
  std::uint32_t status = kSlabHashStatusNotFound;

  table.contains(is_active, key, contains_value, status);
  if (index < count) {
    if (contains != nullptr)
      contains[index] = contains_value;
    if (statuses != nullptr)
      statuses[index] = status;
  }
}

template <class Allocator>
__global__ void
slab_hash_set_erase_kernel(DeviceSlabHashSetU32<Allocator> table,
                           const std::uint32_t *keys, std::uint32_t count,
                           std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_active = index < count;
  std::uint32_t key = is_active ? keys[index] : 0;
  std::uint32_t status = kSlabHashStatusNotFound;

  table.erase(is_active, key, status);
  if (index < count && statuses != nullptr)
    statuses[index] = status;
}

inline std::uint32_t slab_hash_set_thread_count(std::uint32_t operation_count) {
  return operation_count;
}

inline dim3 slab_hash_set_grid(std::uint32_t operation_count,
                               std::uint32_t block_threads = 256) {
  const std::uint32_t threads = slab_hash_set_thread_count(operation_count);
  return dim3((threads + block_threads - 1u) / block_threads);
}

template <class Allocator>
inline cudaError_t
initialize_slab_hash_set(DeviceSlabHashSetU32<Allocator> table,
                         cudaStream_t stream = nullptr) {
  constexpr std::uint32_t kBlockThreads = 256;
  const std::uint32_t total_words =
      slab_hash_slab_storage_count(table.bucket_count, table.slab_capacity) *
      kSlabHashSetWordsPerSlab;
  const dim3 grid((total_words + kBlockThreads - 1u) / kBlockThreads);
  initialize_slab_hash_set_kernel<Allocator>
      <<<grid, kBlockThreads, 0, stream>>>(table);
  return cudaGetLastError();
}

template <class Allocator>
inline cudaError_t
slab_hash_set_insert_batch(DeviceSlabHashSetU32<Allocator> table,
                           const std::uint32_t *keys, std::uint32_t count,
                           std::uint32_t *statuses = nullptr,
                           cudaStream_t stream = nullptr) {
  if (count == 0)
    return cudaSuccess;
  constexpr std::uint32_t kBlockThreads = 256;
  slab_hash_set_insert_kernel<Allocator>
      <<<slab_hash_set_grid(count, kBlockThreads), kBlockThreads, 0, stream>>>(
          table, keys, count, statuses);
  return cudaGetLastError();
}

template <class Allocator>
inline cudaError_t slab_hash_set_contains_batch(
    DeviceSlabHashSetU32<Allocator> table, const std::uint32_t *keys,
    std::uint32_t *contains, std::uint32_t count,
    std::uint32_t *statuses = nullptr, cudaStream_t stream = nullptr) {
  if (count == 0)
    return cudaSuccess;
  constexpr std::uint32_t kBlockThreads = 256;
  slab_hash_set_contains_kernel<Allocator>
      <<<slab_hash_set_grid(count, kBlockThreads), kBlockThreads, 0, stream>>>(
          table, keys, contains, count, statuses);
  return cudaGetLastError();
}

template <class Allocator>
inline cudaError_t
slab_hash_set_erase_batch(DeviceSlabHashSetU32<Allocator> table,
                          const std::uint32_t *keys, std::uint32_t count,
                          std::uint32_t *statuses = nullptr,
                          cudaStream_t stream = nullptr) {
  if (count == 0)
    return cudaSuccess;
  constexpr std::uint32_t kBlockThreads = 256;
  slab_hash_set_erase_kernel<Allocator>
      <<<slab_hash_set_grid(count, kBlockThreads), kBlockThreads, 0, stream>>>(
          table, keys, count, statuses);
  return cudaGetLastError();
}

} // namespace algo::cuda

#endif
