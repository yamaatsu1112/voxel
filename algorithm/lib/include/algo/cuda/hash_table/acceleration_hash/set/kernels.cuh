#pragma once

#include <algo/cuda/hash_table/acceleration_hash/set/device.cuh>

#ifdef __CUDACC__

#include <cuda_runtime.h>

namespace algo::cuda {

template <std::uint32_t ItemWords, class Allocator>
__global__ void initialize_acceleration_hash_set32_kernel(
    DeviceAccelerationHashSet32<ItemWords, Allocator> table) {
  const std::uint32_t lane = static_cast<std::uint32_t>(threadIdx.x) & 31u;
  const std::uint32_t slab_index =
      (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
  if (slab_index >= table.slab_count)
    return;

  typename DeviceAccelerationHashSet32<ItemWords, Allocator>::Slab *slab =
      &table.slabs[slab_index];
  DeviceAccelerationHashSet32<ItemWords, Allocator>::Slab::initialize(slab,
                                                                      lane);
  Allocator::initialize(table.allocator, table.bucket_count);
}

template <std::uint32_t ItemWords, class Allocator>
__global__ void acceleration_hash_set32_find_kernel(
    DeviceAccelerationHashSet32<ItemWords, Allocator> table,
    const std::uint32_t *items, std::uint32_t count, std::uint32_t *ptrs,
    std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_active = index < count;
  const std::uint32_t *item =
      is_active ? items + index * ItemWords : nullptr;
  std::uint32_t ptr = kAccelerationHashNullPtr;
  std::uint32_t status = kAccelerationHashStatusNotFound;

  table.find(is_active, item, ptr, status);
  if (index < count) {
    if (ptrs != nullptr)
      ptrs[index] = ptr;
    if (statuses != nullptr)
      statuses[index] = status;
  }
}

template <std::uint32_t ItemWords, class Allocator>
__global__ void acceleration_hash_set32_insert_unique_unchecked_kernel(
    DeviceAccelerationHashSet32<ItemWords, Allocator> table,
    const std::uint32_t *items, std::uint32_t count, std::uint32_t *ptrs,
    std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_active = index < count;
  const std::uint32_t *item =
      is_active ? items + index * ItemWords : nullptr;
  std::uint32_t ptr = kAccelerationHashNullPtr;
  std::uint32_t status = kAccelerationHashStatusNotFound;

  table.insert_unique_unchecked(is_active, item, ptr, status);
  if (index < count) {
    if (ptrs != nullptr)
      ptrs[index] = ptr;
    if (statuses != nullptr)
      statuses[index] = status;
  }
}

template <std::uint32_t ItemWords, class Allocator>
inline cudaError_t initialize_acceleration_hash_set32(
    DeviceAccelerationHashSet32<ItemWords, Allocator> table,
    cudaStream_t stream = nullptr) {
  constexpr std::uint32_t kBlockThreads = 256;
  const std::uint32_t total_threads = table.slab_count * 32u;
  const dim3 grid((total_threads + kBlockThreads - 1u) / kBlockThreads);
  initialize_acceleration_hash_set32_kernel<ItemWords, Allocator>
      <<<grid, kBlockThreads, 0, stream>>>(table);
  return cudaGetLastError();
}

template <std::uint32_t ItemWords, class Allocator>
inline cudaError_t acceleration_hash_set32_find_batch(
    DeviceAccelerationHashSet32<ItemWords, Allocator> table,
    const std::uint32_t *items, std::uint32_t count, std::uint32_t *ptrs,
    std::uint32_t *statuses = nullptr, cudaStream_t stream = nullptr) {
  if (count == 0)
    return cudaSuccess;
  constexpr std::uint32_t kBlockThreads = 256;
  acceleration_hash_set32_find_kernel<ItemWords, Allocator>
      <<<acceleration_hash::detail::launch_grid(count, kBlockThreads),
         kBlockThreads, 0,
         stream>>>(table, items, count, ptrs, statuses);
  return cudaGetLastError();
}

template <std::uint32_t ItemWords, class Allocator>
inline cudaError_t acceleration_hash_set32_insert_unique_unchecked_batch(
    DeviceAccelerationHashSet32<ItemWords, Allocator> table,
    const std::uint32_t *items, std::uint32_t count, std::uint32_t *ptrs,
    std::uint32_t *statuses = nullptr, cudaStream_t stream = nullptr) {
  if (count == 0)
    return cudaSuccess;
  constexpr std::uint32_t kBlockThreads = 256;
  acceleration_hash_set32_insert_unique_unchecked_kernel<ItemWords, Allocator>
      <<<acceleration_hash::detail::launch_grid(count, kBlockThreads),
         kBlockThreads, 0,
         stream>>>(table, items, count, ptrs, statuses);
  return cudaGetLastError();
}

} // namespace algo::cuda

#endif
