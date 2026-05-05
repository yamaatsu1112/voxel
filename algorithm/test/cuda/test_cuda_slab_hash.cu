#include "cuda_test_utils.cuh"

#include <algo/cuda/slab_hash.cuh>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using DeviceSlabHashMapU32 =
    algo::cuda::DeviceSlabHashMapU32<algo::cuda::BumpSlabAllocator>;
using algo::cuda::SlabHashMapU32Slab;

using cuda_test::has_cuda_device;
using cuda_test::managed_alloc;
using cuda_test::sync_cuda;

void init_slab_hash(DeviceSlabHashMapU32 table) {
  ASSERT_EQ(algo::cuda::initialize_slab_hash_map(table), cudaSuccess);
  sync_cuda();
}

struct ManagedTable {
  explicit ManagedTable(std::uint32_t bucket_count, std::uint32_t slab_capacity)
      : bucket_count(bucket_count), slab_capacity(slab_capacity) {
    slabs = managed_alloc<SlabHashMapU32Slab>(bucket_count + slab_capacity);
    next_slab = managed_alloc<std::uint32_t>(1);
    EXPECT_NE(slabs, nullptr);
    EXPECT_NE(next_slab, nullptr);
  }

  ManagedTable(const ManagedTable &) = delete;
  ManagedTable &operator=(const ManagedTable &) = delete;

  ~ManagedTable() {
    cudaFree(next_slab);
    cudaFree(slabs);
  }

  [[nodiscard]] DeviceSlabHashMapU32 view() const {
    return DeviceSlabHashMapU32{
        slabs, bucket_count, slab_capacity,
        algo::cuda::BumpSlabAllocator::Device{next_slab, slab_capacity}};
  }

  SlabHashMapU32Slab *slabs = nullptr;
  std::uint32_t *next_slab = nullptr;
  std::uint32_t bucket_count = 0;
  std::uint32_t slab_capacity = 0;
};

struct WrappedBumpSlabAllocator {
  using Device = algo::cuda::BumpSlabAllocator::Device;
  using Host = algo::cuda::BumpSlabAllocator::Host;

  __device__ static void initialize(Device allocator) {
    algo::cuda::BumpSlabAllocator::initialize(allocator);
  }

  __device__ static std::uint32_t allocate(Device allocator,
                                           std::uint32_t lane) {
    return algo::cuda::BumpSlabAllocator::allocate(allocator, lane);
  }

  __device__ static void deallocate(Device allocator,
                                    std::uint32_t dynamic_slab_ordinal,
                                    std::uint32_t lane) {
    algo::cuda::BumpSlabAllocator::deallocate(allocator, dynamic_slab_ordinal,
                                              lane);
  }
};

void insert(DeviceSlabHashMapU32 table, const std::uint32_t *keys,
            const std::uint32_t *values, std::uint32_t count,
            std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::slab_hash_map_insert_or_replace_batch(
                table, keys, values, count, statuses),
            cudaSuccess);
  sync_cuda();
}

void insert_or_replace(DeviceSlabHashMapU32 table, const std::uint32_t *keys,
                       const std::uint32_t *values, std::uint32_t count,
                       std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::slab_hash_map_insert_or_replace_batch(
                table, keys, values, count, statuses),
            cudaSuccess);
  sync_cuda();
}

void find(DeviceSlabHashMapU32 table, const std::uint32_t *keys,
          std::uint32_t *values, std::uint32_t count, std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::slab_hash_map_find_batch(table, keys, values, count,
                                                 statuses),
            cudaSuccess);
  sync_cuda();
}

void erase_all(DeviceSlabHashMapU32 table, const std::uint32_t *keys,
               std::uint32_t count, std::uint32_t *statuses) {
  ASSERT_EQ(
      algo::cuda::slab_hash_map_erase_all_batch(table, keys, count, statuses),
      cudaSuccess);
  sync_cuda();
}

__global__ void custom_insert_or_replace_kernel(DeviceSlabHashMapU32 table,
                                                const std::uint32_t *keys,
                                                const std::uint32_t *values,
                                                std::uint32_t count,
                                                std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = index < count;
  std::uint32_t key = active ? keys[index] : 0;
  std::uint32_t value = active ? values[index] : 0;
  std::uint32_t status = algo::cuda::kSlabHashStatusNotFound;

  table.insert_or_replace(active, key, value, status);

  if (index < count)
    statuses[index] = status;
}

__global__ void custom_find_kernel(DeviceSlabHashMapU32 table,
                                   const std::uint32_t *keys,
                                   std::uint32_t *values, std::uint32_t count,
                                   std::uint32_t *statuses) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = index < count;
  std::uint32_t key = active ? keys[index] : 0;
  std::uint32_t value = algo::cuda::kSlabHashNotFoundValue;
  std::uint32_t status = algo::cuda::kSlabHashStatusNotFound;

  table.find(active, key, value, status);

  if (index < count) {
    values[index] = value;
    statuses[index] = status;
  }
}

void custom_insert(DeviceSlabHashMapU32 table, const std::uint32_t *keys,
                   const std::uint32_t *values, std::uint32_t count,
                   std::uint32_t *statuses) {
  custom_insert_or_replace_kernel<<<1, 64>>>(table, keys, values, count,
                                             statuses);
  sync_cuda();
}

void custom_find(DeviceSlabHashMapU32 table, const std::uint32_t *keys,
                 std::uint32_t *values, std::uint32_t count,
                 std::uint32_t *statuses) {
  custom_find_kernel<<<1, 64>>>(table, keys, values, count, statuses);
  sync_cuda();
}

template <class Map> void run_host_map_clear_scenario() {
  constexpr std::uint32_t kCount = algo::cuda::kSlabHashMapSlotsPerSlab + 1u;
  Map table(1, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    keys[i] = i + 1u;
    values[i] = 100u + i;
  }

  ASSERT_EQ(table.insert_or_replace_batch(keys, values, kCount, statuses),
            cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);

  ASSERT_EQ(table.find_batch(keys, results, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], values[i]);
  }

  table.clear();

  ASSERT_EQ(table.find_batch(keys, results, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusNotFound);
    EXPECT_EQ(results[i], algo::cuda::kSlabHashNotFoundValue);
  }

  for (std::uint32_t i = 0; i < kCount; ++i)
    values[i] = 500u + i;

  ASSERT_EQ(table.insert_or_replace_batch(keys, values, kCount, statuses),
            cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);

  ASSERT_EQ(table.find_batch(keys, results, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], values[i]);
  }

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, InsertAndFindU32Keys) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 4;
  ManagedTable table(8, 8);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    keys[i] = 10u + i * 7u;
    values[i] = 100u + i;
  }

  init_slab_hash(table.view());
  insert(table.view(), keys, values, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);
  }

  find(table.view(), keys, results, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], values[i]);
  }

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, DeviceApiCanBeCalledFromCustomKernel) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 32;
  ManagedTable table(8, 8);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    keys[i] = 1000u + i;
    values[i] = 2000u + i;
  }

  init_slab_hash(table.view());
  custom_insert(table.view(), keys, values, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);
  }

  custom_find(table.view(), keys, results, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], values[i]);
  }

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, SupportsAllocatorPolicyParameter) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 4;
  algo::cuda::BasicSlabHashMapU32<WrappedBumpSlabAllocator> table(4, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    keys[i] = 500u + i;
    values[i] = 700u + i;
  }

  ASSERT_EQ(table.insert_or_replace_batch(keys, values, kCount, statuses),
            cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);
  }

  ASSERT_EQ(table.find_batch(keys, results, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], values[i]);
  }

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, DefaultHostApiUsesSlabAllocAndClearResetsAllocator) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  run_host_map_clear_scenario<algo::cuda::SlabHashMapU32>();
}

TEST(CudaSlabHash, ExplicitBumpAllocatorHostApiRemainsSupported) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  run_host_map_clear_scenario<
      algo::cuda::BasicSlabHashMapU32<algo::cuda::BumpSlabAllocator>>();
}

TEST(CudaSlabHash, MissingKeysReturnNotFound) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 3;
  ManagedTable table(4, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(statuses, nullptr);

  keys[0] = 101;
  keys[1] = 202;
  keys[2] = 303;

  init_slab_hash(table.view());
  find(table.view(), keys, values, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusNotFound);
    EXPECT_EQ(values[i], algo::cuda::kSlabHashNotFoundValue);
  }

  cudaFree(statuses);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, GrowsCollisionChain) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = algo::cuda::kSlabHashMapSlotsPerSlab + 5u;
  ManagedTable table(1, 4);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    keys[i] = i + 1u;
    values[i] = 1000u + i;
  }

  init_slab_hash(table.view());
  insert(table.view(), keys, values, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);
  }
  EXPECT_GT(*table.next_slab, 0u);

  find(table.view(), keys, results, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], values[i]);
  }

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, InsertOrReplaceUpdatesExistingValue) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  ManagedTable table(4, 2);
  auto *keys = managed_alloc<std::uint32_t>(1);
  auto *values = managed_alloc<std::uint32_t>(1);
  auto *results = managed_alloc<std::uint32_t>(1);
  auto *statuses = managed_alloc<std::uint32_t>(1);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  init_slab_hash(table.view());
  keys[0] = 42;
  values[0] = 100;
  insert_or_replace(table.view(), keys, values, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusInserted);

  values[0] = 200;
  insert_or_replace(table.view(), keys, values, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusReplaced);

  find(table.view(), keys, results, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusFound);
  EXPECT_EQ(results[0], 200u);

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, InsertOrReplaceUsesFirstAvailableOrMatchingSlot) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = algo::cuda::kSlabHashMapSlotsPerSlab + 1u;
  ManagedTable table(1, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *result = managed_alloc<std::uint32_t>(1);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(result, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    keys[i] = i + 1u;
    values[i] = 100u + i;
  }

  init_slab_hash(table.view());
  insert(table.view(), keys, values, kCount, statuses);
  EXPECT_GT(*table.next_slab, 0u);

  erase_all(table.view(), keys, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusErased);

  keys[0] = kCount;
  values[0] = 999;
  insert_or_replace(table.view(), keys, values, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusInserted);

  find(table.view(), keys, result, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusFound);
  EXPECT_EQ(result[0], 999u);

  erase_all(table.view(), keys, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusErased);

  find(table.view(), keys, result, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusNotFound);
  EXPECT_EQ(result[0], algo::cuda::kSlabHashNotFoundValue);

  cudaFree(statuses);
  cudaFree(result);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, DuplicateKeysInSameBatchReplaceOneSlot) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 8;
  ManagedTable table(4, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *query = managed_alloc<std::uint32_t>(1);
  auto *result = managed_alloc<std::uint32_t>(1);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(query, nullptr);
  ASSERT_NE(result, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    keys[i] = 99;
    values[i] = 1000u + i;
  }
  query[0] = 99;

  init_slab_hash(table.view());
  insert(table.view(), keys, values, kCount, statuses);

  std::uint32_t inserted = 0;
  std::uint32_t replaced = 0;
  for (std::uint32_t i = 0; i < kCount; ++i) {
    if (statuses[i] == algo::cuda::kSlabHashStatusInserted)
      ++inserted;
    if (statuses[i] == algo::cuda::kSlabHashStatusReplaced) {
      ++replaced;
    }
  }
  EXPECT_EQ(inserted, 1u);
  EXPECT_EQ(replaced, kCount - 1u);

  find(table.view(), query, result, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusFound);
  EXPECT_EQ(result[0], 1000u + kCount - 1u);

  cudaFree(statuses);
  cudaFree(result);
  cudaFree(query);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, EraseAllMarksKeyDeleted) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 2;
  ManagedTable table(4, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  keys[0] = 7;
  keys[1] = 11;
  values[0] = 70;
  values[1] = 110;

  init_slab_hash(table.view());
  insert(table.view(), keys, values, kCount, statuses);

  erase_all(table.view(), keys, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusErased);

  find(table.view(), keys, results, kCount, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusNotFound);
  EXPECT_EQ(results[0], algo::cuda::kSlabHashNotFoundValue);
  EXPECT_EQ(statuses[1], algo::cuda::kSlabHashStatusFound);
  EXPECT_EQ(results[1], 110u);

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(values);
  cudaFree(keys);
}

TEST(CudaSlabHash, ReportsOverflowWhenSlabPoolIsFull) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = algo::cuda::kSlabHashMapSlotsPerSlab + 1u;
  ManagedTable table(1, 0);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *values = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i) {
    keys[i] = i + 1u;
    values[i] = i + 100u;
  }

  init_slab_hash(table.view());
  insert(table.view(), keys, values, kCount, statuses);

  std::uint32_t inserted = 0;
  std::uint32_t overflow = 0;
  for (std::uint32_t i = 0; i < kCount; ++i) {
    if (statuses[i] == algo::cuda::kSlabHashStatusInserted)
      ++inserted;
    if (statuses[i] == algo::cuda::kSlabHashStatusOverflow)
      ++overflow;
  }
  EXPECT_EQ(inserted, algo::cuda::kSlabHashMapSlotsPerSlab);
  EXPECT_EQ(overflow, 1u);

  cudaFree(statuses);
  cudaFree(values);
  cudaFree(keys);
}

} // namespace
