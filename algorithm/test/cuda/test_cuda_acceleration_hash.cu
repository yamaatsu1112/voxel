#include "cuda_test_utils.cuh"

#include <algo/cuda/acceleration_hash.cuh>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

template <std::uint32_t ItemWords>
using DeviceAccelerationHashSet32 =
    algo::cuda::DeviceAccelerationHashSet32<ItemWords,
                                            algo::cuda::BumpSlabAllocator>;

template <std::uint32_t ItemWords>
using DeviceAccelerationHashMap32 =
    algo::cuda::DeviceAccelerationHashMap32<ItemWords,
                                            algo::cuda::BumpSlabAllocator>;

using cuda_test::has_cuda_device;
using cuda_test::managed_alloc;
using cuda_test::sync_cuda;

template <std::uint32_t ItemWords> struct ManagedTable {
  using Slab = algo::cuda::AccelerationHashSlab32<ItemWords>;

  explicit ManagedTable(std::uint32_t bucket_count,
                        std::uint32_t overflow_slab_capacity)
      : bucket_count(bucket_count),
        overflow_slab_capacity(overflow_slab_capacity),
        slab_count(bucket_count + overflow_slab_capacity) {
    slabs = managed_alloc<Slab>(slab_count);
    next_slab = managed_alloc<std::uint32_t>(1);
    if (slab_count != 0)
      EXPECT_NE(slabs, nullptr);
    EXPECT_NE(next_slab, nullptr);
  }

  ManagedTable(const ManagedTable &) = delete;
  ManagedTable &operator=(const ManagedTable &) = delete;

  ~ManagedTable() {
    cudaFree(next_slab);
    cudaFree(slabs);
  }

  [[nodiscard]] DeviceAccelerationHashSet32<ItemWords> view() const {
    return DeviceAccelerationHashSet32<ItemWords>{
        slabs, bucket_count, slab_count,
        algo::cuda::BumpSlabAllocator::Device{next_slab}};
  }

  Slab *slabs = nullptr;
  std::uint32_t *next_slab = nullptr;
  std::uint32_t bucket_count = 0;
  std::uint32_t overflow_slab_capacity = 0;
  std::uint32_t slab_count = 0;
};

template <std::uint32_t ItemWords> struct ManagedMap {
  using Slab = algo::cuda::AccelerationHashMapSlab32<ItemWords>;

  explicit ManagedMap(std::uint32_t bucket_count,
                      std::uint32_t overflow_slab_capacity)
      : bucket_count(bucket_count),
        overflow_slab_capacity(overflow_slab_capacity),
        slab_count(bucket_count + overflow_slab_capacity) {
    slabs = managed_alloc<Slab>(slab_count);
    next_slab = managed_alloc<std::uint32_t>(1);
    if (slab_count != 0)
      EXPECT_NE(slabs, nullptr);
    EXPECT_NE(next_slab, nullptr);
  }

  ManagedMap(const ManagedMap &) = delete;
  ManagedMap &operator=(const ManagedMap &) = delete;

  ~ManagedMap() {
    cudaFree(next_slab);
    cudaFree(slabs);
  }

  [[nodiscard]] DeviceAccelerationHashMap32<ItemWords> view() const {
    return DeviceAccelerationHashMap32<ItemWords>{
        slabs, bucket_count, slab_count,
        algo::cuda::BumpSlabAllocator::Device{next_slab}};
  }

  Slab *slabs = nullptr;
  std::uint32_t *next_slab = nullptr;
  std::uint32_t bucket_count = 0;
  std::uint32_t overflow_slab_capacity = 0;
  std::uint32_t slab_count = 0;
};

template <std::uint32_t ItemWords>
void init_acceleration_hash(DeviceAccelerationHashSet32<ItemWords> table) {
  ASSERT_EQ(algo::cuda::initialize_acceleration_hash_set32(table), cudaSuccess);
  sync_cuda();
}

template <std::uint32_t ItemWords>
void init_acceleration_hash(DeviceAccelerationHashMap32<ItemWords> table) {
  ASSERT_EQ(algo::cuda::initialize_acceleration_hash_map32(table), cudaSuccess);
  sync_cuda();
}

template <std::uint32_t ItemWords>
void insert_unique_unchecked(DeviceAccelerationHashSet32<ItemWords> table,
                   const std::uint32_t *items, std::uint32_t count,
                   std::uint32_t *ptrs, std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::acceleration_hash_set32_insert_unique_unchecked_batch(
                table, items, count, ptrs, statuses),
            cudaSuccess);
  sync_cuda();
}

template <std::uint32_t ItemWords>
void insert_unique_unchecked(DeviceAccelerationHashMap32<ItemWords> table,
                             const std::uint32_t *items,
                             const std::uint32_t *values,
                             std::uint32_t count, std::uint32_t *ptrs,
                             std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::acceleration_hash_map32_insert_unique_unchecked_batch(
                table, items, values, count, ptrs, statuses),
            cudaSuccess);
  sync_cuda();
}

template <std::uint32_t ItemWords>
void find(DeviceAccelerationHashSet32<ItemWords> table,
          const std::uint32_t *items, std::uint32_t count, std::uint32_t *ptrs,
          std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::acceleration_hash_set32_find_batch(table, items, count,
                                                           ptrs, statuses),
            cudaSuccess);
  sync_cuda();
}

template <std::uint32_t ItemWords>
void find(DeviceAccelerationHashMap32<ItemWords> table,
          const std::uint32_t *items, std::uint32_t count, std::uint32_t *ptrs,
          std::uint32_t *values, std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::acceleration_hash_map32_find_batch(
                table, items, count, ptrs, values, statuses),
            cudaSuccess);
  sync_cuda();
}

template <std::uint32_t ItemWords>
void fill_items(std::uint32_t *items, std::uint32_t count,
                std::uint32_t seed = 0) {
  for (std::uint32_t i = 0; i < count; ++i) {
    for (std::uint32_t word = 0; word < ItemWords; ++word) {
      items[i * ItemWords + word] =
          0x10000u + seed * 97u + i * 131u + word * 17u;
    }
  }
}

template <std::uint32_t ItemWords>
__global__ void
custom_map_insert_unique_unchecked_kernel(
    DeviceAccelerationHashMap32<ItemWords> table, const std::uint32_t *items,
    const std::uint32_t *values, std::uint32_t count, std::uint32_t *ptrs,
    std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = index < count;
  const std::uint32_t *item = active ? items + index * ItemWords : nullptr;
  const std::uint32_t value = active ? values[index] : 0u;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;

  table.insert_unique_unchecked(active, item, value, ptr, status);

  if (index < count) {
    ptrs[index] = ptr;
    statuses[index] = status;
  }
}

template <std::uint32_t ItemWords>
__global__ void custom_map_find_kernel(DeviceAccelerationHashMap32<ItemWords> table,
                                       const std::uint32_t *items,
                                       std::uint32_t count,
                                       std::uint32_t *ptrs,
                                       std::uint32_t *values,
                                       std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = index < count;
  const std::uint32_t *item = active ? items + index * ItemWords : nullptr;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t value = 0;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;

  table.find(active, item, ptr, value, status);

  if (index < count) {
    ptrs[index] = ptr;
    values[index] = value;
    statuses[index] = status;
  }
}

template <std::uint32_t ItemWords>
__global__ void
custom_insert_kernel(DeviceAccelerationHashSet32<ItemWords> table,
                     const std::uint32_t *items, std::uint32_t count,
                     std::uint32_t *ptrs, std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = index < count;
  const std::uint32_t *item = active ? items + index * ItemWords : nullptr;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;

  table.insert_unique_unchecked(active, item, ptr, status);

  if (index < count) {
    ptrs[index] = ptr;
    statuses[index] = status;
  }
}

template <std::uint32_t ItemWords>
__global__ void custom_find_kernel(DeviceAccelerationHashSet32<ItemWords> table,
                                   const std::uint32_t *items,
                                   std::uint32_t count, std::uint32_t *ptrs,
                                   std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = index < count;
  const std::uint32_t *item = active ? items + index * ItemWords : nullptr;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;

  table.find(active, item, ptr, status);

  if (index < count) {
    ptrs[index] = ptr;
    statuses[index] = status;
  }
}

template <std::uint32_t ItemWords>
__global__ void read_items_kernel(DeviceAccelerationHashSet32<ItemWords> table,
                                  const std::uint32_t *ptrs,
                                  std::uint32_t count, std::uint32_t *out_items,
                                  std::uint32_t *ok) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  std::uint32_t item[ItemWords];
  const bool found = table.read_item(ptrs[index], item);
  ok[index] = found ? 1u : 0u;
  if (found) {
    for (std::uint32_t word = 0; word < ItemWords; ++word)
      out_items[index * ItemWords + word] = item[word];
  }
}

template <std::uint32_t ItemWords>
void expect_inserted_and_found(const std::uint32_t *insert_ptrs,
                               const std::uint32_t *find_ptrs,
                               const std::uint32_t *statuses,
                               std::uint32_t count) {
  for (std::uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusFound);
    EXPECT_NE(find_ptrs[i], algo::cuda::kAccelerationHashNullPtr);
    EXPECT_EQ(find_ptrs[i], insert_ptrs[i]);
  }
}

TEST(CudaAccelerationHash, EmptyTableFindsNothing) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 2;
  constexpr std::uint32_t kCount = 4;
  ManagedTable<kItemWords> table(8, 1);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(ptrs, nullptr);
  ASSERT_NE(statuses, nullptr);
  fill_items<kItemWords>(items, kCount);

  init_acceleration_hash(table.view());
  find<kItemWords>(table.view(), items, kCount, ptrs, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusNotFound);
    EXPECT_EQ(ptrs[i], algo::cuda::kAccelerationHashNullPtr);
  }

  cudaFree(statuses);
  cudaFree(ptrs);
  cudaFree(items);
}

TEST(CudaAccelerationHash, InsertAndFindLargeItems) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 6;
  constexpr std::uint32_t kCount = 16;
  ManagedTable<kItemWords> table(8, 2);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *insert_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *find_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(insert_ptrs, nullptr);
  ASSERT_NE(find_ptrs, nullptr);
  ASSERT_NE(statuses, nullptr);
  fill_items<kItemWords>(items, kCount);

  init_acceleration_hash(table.view());
  insert_unique_unchecked<kItemWords>(table.view(), items, kCount, insert_ptrs,
                                      statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusInserted);
    EXPECT_NE(insert_ptrs[i], algo::cuda::kAccelerationHashNullPtr);
  }

  find<kItemWords>(table.view(), items, kCount, find_ptrs, statuses);
  expect_inserted_and_found<kItemWords>(insert_ptrs, find_ptrs, statuses,
                                        kCount);

  cudaFree(statuses);
  cudaFree(find_ptrs);
  cudaFree(insert_ptrs);
  cudaFree(items);
}

TEST(CudaAccelerationHash, DeviceApiCanBeCalledFromCustomKernel) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 10;
  constexpr std::uint32_t kCount = 64;
  ManagedTable<kItemWords> table(16, 4);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *insert_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *find_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(insert_ptrs, nullptr);
  ASSERT_NE(find_ptrs, nullptr);
  ASSERT_NE(statuses, nullptr);
  fill_items<kItemWords>(items, kCount);

  init_acceleration_hash(table.view());
  custom_insert_kernel<kItemWords>
      <<<1, 96>>>(table.view(), items, kCount, insert_ptrs, statuses);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusInserted);

  custom_find_kernel<kItemWords>
      <<<1, 96>>>(table.view(), items, kCount, find_ptrs, statuses);
  sync_cuda();
  expect_inserted_and_found<kItemWords>(insert_ptrs, find_ptrs, statuses,
                                        kCount);

  cudaFree(statuses);
  cudaFree(find_ptrs);
  cudaFree(insert_ptrs);
  cudaFree(items);
}

TEST(CudaAccelerationHash, GrowsCollisionChain) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 2;
  constexpr std::uint32_t kCount =
      algo::cuda::kAccelerationHashSlotsPerSlab + 5u;
  ManagedTable<kItemWords> table(1, 2);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *insert_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *find_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(insert_ptrs, nullptr);
  ASSERT_NE(find_ptrs, nullptr);
  ASSERT_NE(statuses, nullptr);
  fill_items<kItemWords>(items, kCount);

  init_acceleration_hash(table.view());
  insert_unique_unchecked<kItemWords>(table.view(), items, kCount, insert_ptrs,
                                      statuses);
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusInserted);
  EXPECT_GT(*table.next_slab, table.bucket_count);
  EXPECT_EQ(table.slabs[0].next, table.bucket_count);

  find<kItemWords>(table.view(), items, kCount, find_ptrs, statuses);
  expect_inserted_and_found<kItemWords>(insert_ptrs, find_ptrs, statuses,
                                        kCount);

  cudaFree(statuses);
  cudaFree(find_ptrs);
  cudaFree(insert_ptrs);
  cudaFree(items);
}

TEST(CudaAccelerationHash, ReportsOverflowWhenNoSlabCanGrow) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 2;
  constexpr std::uint32_t kCount =
      algo::cuda::kAccelerationHashSlotsPerSlab + 1u;
  ManagedTable<kItemWords> table(1, 0);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(ptrs, nullptr);
  ASSERT_NE(statuses, nullptr);
  fill_items<kItemWords>(items, kCount);

  init_acceleration_hash(table.view());
  insert_unique_unchecked<kItemWords>(table.view(), items, kCount, ptrs,
                                      statuses);

  std::uint32_t inserted = 0;
  std::uint32_t overflow = 0;
  for (std::uint32_t i = 0; i < kCount; ++i) {
    if (statuses[i] == algo::cuda::kAccelerationHashStatusInserted)
      ++inserted;
    if (statuses[i] == algo::cuda::kAccelerationHashStatusOverflow)
      ++overflow;
  }
  EXPECT_EQ(inserted, algo::cuda::kAccelerationHashSlotsPerSlab);
  EXPECT_EQ(overflow, 1u);

  cudaFree(statuses);
  cudaFree(ptrs);
  cudaFree(items);
}

TEST(CudaAccelerationHash, FindComparesFullPayload) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 2;
  ManagedTable<kItemWords> table(4, 1);
  auto *items = managed_alloc<std::uint32_t>(2 * kItemWords);
  auto *ptrs = managed_alloc<std::uint32_t>(2);
  auto *statuses = managed_alloc<std::uint32_t>(2);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(ptrs, nullptr);
  ASSERT_NE(statuses, nullptr);

  items[0] = 1;
  items[1] = 2;
  items[2] = 1;
  items[3] = 3;

  init_acceleration_hash(table.view());
  insert_unique_unchecked<kItemWords>(table.view(), items, 1, ptrs, statuses);
  ASSERT_EQ(statuses[0], algo::cuda::kAccelerationHashStatusInserted);

  find<kItemWords>(table.view(), items + kItemWords, 1, ptrs, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kAccelerationHashStatusNotFound);
  EXPECT_EQ(ptrs[0], algo::cuda::kAccelerationHashNullPtr);

  cudaFree(statuses);
  cudaFree(ptrs);
  cudaFree(items);
}

TEST(CudaAccelerationHash, ReadsPayloadFromReturnedPointer) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 6;
  constexpr std::uint32_t kCount = 8;
  ManagedTable<kItemWords> table(4, 2);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *read_items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  auto *ok = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(read_items, nullptr);
  ASSERT_NE(ptrs, nullptr);
  ASSERT_NE(statuses, nullptr);
  ASSERT_NE(ok, nullptr);
  fill_items<kItemWords>(items, kCount);

  init_acceleration_hash(table.view());
  insert_unique_unchecked<kItemWords>(table.view(), items, kCount, ptrs,
                                      statuses);
  read_items_kernel<kItemWords>
      <<<1, 32>>>(table.view(), ptrs, kCount, read_items, ok);
  sync_cuda();

  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(ok[i], 1u);
    for (std::uint32_t word = 0; word < kItemWords; ++word) {
      EXPECT_EQ(read_items[i * kItemWords + word],
                items[i * kItemWords + word]);
    }
  }

  cudaFree(ok);
  cudaFree(statuses);
  cudaFree(ptrs);
  cudaFree(read_items);
  cudaFree(items);
}

TEST(CudaAccelerationHash, HostApiInsertAndFind) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 2;
  constexpr std::uint32_t kCount = 5;
  algo::cuda::AccelerationHashSet32<kItemWords> table(4, 2);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *insert_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *find_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(insert_ptrs, nullptr);
  ASSERT_NE(find_ptrs, nullptr);
  ASSERT_NE(statuses, nullptr);
  fill_items<kItemWords>(items, kCount);

  ASSERT_EQ(
      table.insert_unique_unchecked_batch(items, kCount, insert_ptrs, statuses),
      cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusInserted);

  ASSERT_EQ(table.find_batch(items, kCount, find_ptrs, statuses), cudaSuccess);
  sync_cuda();
  expect_inserted_and_found<kItemWords>(insert_ptrs, find_ptrs, statuses,
                                        kCount);

  cudaFree(statuses);
  cudaFree(find_ptrs);
  cudaFree(insert_ptrs);
  cudaFree(items);
}

TEST(CudaAccelerationHashMap, InsertAndFindValues) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 8;
  constexpr std::uint32_t kCount = 16;
  ManagedMap<kItemWords> table(8, 2);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *insert_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *find_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *found_values = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(insert_ptrs, nullptr);
  ASSERT_NE(find_ptrs, nullptr);
  ASSERT_NE(found_values, nullptr);
  ASSERT_NE(statuses, nullptr);
  fill_items<kItemWords>(items, kCount);
  for (std::uint32_t i = 0; i < kCount; ++i)
    values[i] = 1000u + i;

  init_acceleration_hash(table.view());
  insert_unique_unchecked<kItemWords>(table.view(), items, values, kCount,
                                      insert_ptrs, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusInserted);
    EXPECT_NE(insert_ptrs[i], algo::cuda::kAccelerationHashNullPtr);
  }

  find<kItemWords>(table.view(), items, kCount, find_ptrs, found_values,
                   statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusFound);
    EXPECT_EQ(find_ptrs[i], insert_ptrs[i]);
    EXPECT_EQ(found_values[i], values[i]);
  }

  cudaFree(statuses);
  cudaFree(found_values);
  cudaFree(find_ptrs);
  cudaFree(insert_ptrs);
  cudaFree(values);
  cudaFree(items);
}

TEST(CudaAccelerationHashMap, DuplicateInsertSearchesSingleRepresentative) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 8;
  constexpr std::uint32_t kCount = 4;
  ManagedMap<kItemWords> table(1, 1);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *insert_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *find_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *found_values = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(insert_ptrs, nullptr);
  ASSERT_NE(find_ptrs, nullptr);
  ASSERT_NE(found_values, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    for (std::uint32_t word = 0; word < kItemWords; ++word)
      items[i * kItemWords + word] = 0xabc000u + word * 17u;
    values[i] = 10u + i;
  }

  init_acceleration_hash(table.view());
  insert_unique_unchecked<kItemWords>(table.view(), items, values, kCount,
                                      insert_ptrs, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusInserted);

  find<kItemWords>(table.view(), items, kCount, find_ptrs, found_values,
                   statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusFound);
    EXPECT_EQ(find_ptrs[i], find_ptrs[0]);
    EXPECT_EQ(found_values[i], found_values[0]);
  }
  EXPECT_GE(found_values[0], values[0]);
  EXPECT_LE(found_values[0], values[kCount - 1u]);

  cudaFree(statuses);
  cudaFree(found_values);
  cudaFree(find_ptrs);
  cudaFree(insert_ptrs);
  cudaFree(values);
  cudaFree(items);
}

TEST(CudaAccelerationHashMap, DeviceApiCanBeCalledFromCustomKernel) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kItemWords = 8;
  constexpr std::uint32_t kCount = 64;
  ManagedMap<kItemWords> table(16, 4);
  auto *items = managed_alloc<std::uint32_t>(kCount * kItemWords);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *insert_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *find_ptrs = managed_alloc<std::uint32_t>(kCount);
  auto *found_values = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(items, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(insert_ptrs, nullptr);
  ASSERT_NE(find_ptrs, nullptr);
  ASSERT_NE(found_values, nullptr);
  ASSERT_NE(statuses, nullptr);
  fill_items<kItemWords>(items, kCount, 7u);
  for (std::uint32_t i = 0; i < kCount; ++i)
    values[i] = 5000u + i * 3u;

  init_acceleration_hash(table.view());
  custom_map_insert_unique_unchecked_kernel<kItemWords>
      <<<1, 96>>>(table.view(), items, values, kCount, insert_ptrs, statuses);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusInserted);

  custom_map_find_kernel<kItemWords><<<1, 96>>>(
      table.view(), items, kCount, find_ptrs, found_values, statuses);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kAccelerationHashStatusFound);
    EXPECT_EQ(find_ptrs[i], insert_ptrs[i]);
    EXPECT_EQ(found_values[i], values[i]);
  }

  cudaFree(statuses);
  cudaFree(found_values);
  cudaFree(find_ptrs);
  cudaFree(insert_ptrs);
  cudaFree(values);
  cudaFree(items);
}

} // namespace
