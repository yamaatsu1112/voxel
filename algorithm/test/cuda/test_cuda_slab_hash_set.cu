#include "cuda_test_utils.cuh"

#include <algo/cuda/slab_hash.cuh>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace {

using DeviceSlabHashSetU32 =
    algo::cuda::DeviceSlabHashSetU32<algo::cuda::BumpSlabAllocator>;
using algo::cuda::SlabHashSetU32Slab;

using cuda_test::has_cuda_device;
using cuda_test::managed_alloc;
using cuda_test::sync_cuda;

struct ManagedSetTable {
  explicit ManagedSetTable(std::uint32_t bucket_count,
                           std::uint32_t slab_capacity)
      : bucket_count(bucket_count), slab_capacity(slab_capacity) {
    slabs = managed_alloc<SlabHashSetU32Slab>(bucket_count + slab_capacity);
    next_slab = managed_alloc<std::uint32_t>(1);
    EXPECT_NE(slabs, nullptr);
    EXPECT_NE(next_slab, nullptr);
  }

  ManagedSetTable(const ManagedSetTable &) = delete;
  ManagedSetTable &operator=(const ManagedSetTable &) = delete;

  ~ManagedSetTable() {
    cudaFree(next_slab);
    cudaFree(slabs);
  }

  [[nodiscard]] DeviceSlabHashSetU32 view() const {
    return DeviceSlabHashSetU32{
        slabs, bucket_count, slab_capacity,
        algo::cuda::BumpSlabAllocator::Device{next_slab, slab_capacity}};
  }

  SlabHashSetU32Slab *slabs = nullptr;
  std::uint32_t *next_slab = nullptr;
  std::uint32_t bucket_count = 0;
  std::uint32_t slab_capacity = 0;
};

void init_slab_hash_set(DeviceSlabHashSetU32 table) {
  ASSERT_EQ(algo::cuda::initialize_slab_hash_set(table), cudaSuccess);
  sync_cuda();
}

void insert(DeviceSlabHashSetU32 table, const std::uint32_t *keys,
            std::uint32_t count, std::uint32_t *statuses) {
  ASSERT_EQ(
      algo::cuda::slab_hash_set_insert_batch(table, keys, count, statuses),
      cudaSuccess);
  sync_cuda();
}

void contains(DeviceSlabHashSetU32 table, const std::uint32_t *keys,
              std::uint32_t *results, std::uint32_t count,
              std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::slab_hash_set_contains_batch(table, keys, results,
                                                     count, statuses),
            cudaSuccess);
  sync_cuda();
}

void erase(DeviceSlabHashSetU32 table, const std::uint32_t *keys,
           std::uint32_t count, std::uint32_t *statuses) {
  ASSERT_EQ(algo::cuda::slab_hash_set_erase_batch(table, keys, count, statuses),
            cudaSuccess);
  sync_cuda();
}

template <class Set> void run_host_set_clear_scenario() {
  constexpr std::uint32_t kCount = algo::cuda::kSlabHashSetKeysPerSlab + 1u;
  Set table(1, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i)
    keys[i] = i + 1u;

  ASSERT_EQ(table.insert_batch(keys, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);

  ASSERT_EQ(table.contains_batch(keys, results, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], 1u);
  }

  table.clear();

  ASSERT_EQ(table.contains_batch(keys, results, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusNotFound);
    EXPECT_EQ(results[i], 0u);
  }

  ASSERT_EQ(table.insert_batch(keys, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);

  ASSERT_EQ(table.contains_batch(keys, results, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], 1u);
  }

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(keys);
}

TEST(CudaSlabHashSet, InsertAndContainsU32Keys) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 4;
  ManagedSetTable table(8, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i)
    keys[i] = 10u + i * 7u;

  init_slab_hash_set(table.view());
  insert(table.view(), keys, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);
  }

  contains(table.view(), keys, results, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], 1u);
  }

  keys[0] = 9999;
  contains(table.view(), keys, results, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusNotFound);
  EXPECT_EQ(results[0], 0u);

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(keys);
}

TEST(CudaSlabHashSet, DefaultHostApiUsesSlabAllocAndClearResetsAllocator) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  run_host_set_clear_scenario<algo::cuda::SlabHashSetU32>();
}

TEST(CudaSlabHashSet, ExplicitBumpAllocatorHostApiRemainsSupported) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  run_host_set_clear_scenario<
      algo::cuda::BasicSlabHashSetU32<algo::cuda::BumpSlabAllocator>>();
}

TEST(CudaSlabHashSet, HostApiInsertContainsAndErase) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 3;
  algo::cuda::SlabHashSetU32 table(4, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  keys[0] = 7;
  keys[1] = 11;
  keys[2] = 13;

  ASSERT_EQ(table.insert_batch(keys, kCount, statuses), cudaSuccess);
  sync_cuda();
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);
  }

  ASSERT_EQ(table.erase_batch(keys, 1, statuses), cudaSuccess);
  sync_cuda();
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusErased);

  ASSERT_EQ(table.contains_batch(keys, results, kCount, statuses), cudaSuccess);
  sync_cuda();
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusNotFound);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(statuses[1], algo::cuda::kSlabHashStatusFound);
  EXPECT_EQ(results[1], 1u);
  EXPECT_EQ(statuses[2], algo::cuda::kSlabHashStatusFound);
  EXPECT_EQ(results[2], 1u);

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(keys);
}

TEST(CudaSlabHashSet, GrowsCollisionChain) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = algo::cuda::kSlabHashSetKeysPerSlab + 5u;
  ManagedSetTable table(1, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i)
    keys[i] = i + 1u;

  init_slab_hash_set(table.view());
  insert(table.view(), keys, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusInserted);
  }
  EXPECT_GT(*table.next_slab, 0u);

  contains(table.view(), keys, results, kCount, statuses);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(statuses[i], algo::cuda::kSlabHashStatusFound);
    EXPECT_EQ(results[i], 1u);
  }

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(keys);
}

TEST(CudaSlabHashSet, DuplicateKeysReportAlreadyPresent) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 8;
  ManagedSetTable table(4, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i)
    keys[i] = 99;

  init_slab_hash_set(table.view());
  insert(table.view(), keys, kCount, statuses);

  std::uint32_t inserted = 0;
  std::uint32_t already_present = 0;
  for (std::uint32_t i = 0; i < kCount; ++i) {
    if (statuses[i] == algo::cuda::kSlabHashStatusInserted)
      ++inserted;
    if (statuses[i] == algo::cuda::kSlabHashStatusAlreadyPresent) {
      ++already_present;
    }
  }
  EXPECT_EQ(inserted, 1u);
  EXPECT_EQ(already_present, kCount - 1u);

  cudaFree(statuses);
  cudaFree(keys);
}

TEST(CudaSlabHashSet, EraseMarksKeyDeletedAndReusesSlot) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = 2;
  ManagedSetTable table(4, 2);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *results = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(results, nullptr);
  ASSERT_NE(statuses, nullptr);

  keys[0] = 7;
  keys[1] = 11;

  init_slab_hash_set(table.view());
  insert(table.view(), keys, kCount, statuses);
  erase(table.view(), keys, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusErased);

  contains(table.view(), keys, results, kCount, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusNotFound);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(statuses[1], algo::cuda::kSlabHashStatusFound);
  EXPECT_EQ(results[1], 1u);

  insert(table.view(), keys, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusInserted);
  contains(table.view(), keys, results, 1, statuses);
  EXPECT_EQ(statuses[0], algo::cuda::kSlabHashStatusFound);
  EXPECT_EQ(results[0], 1u);

  cudaFree(statuses);
  cudaFree(results);
  cudaFree(keys);
}

TEST(CudaSlabHashSet, ReportsOverflowWhenSlabPoolIsFull) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCount = algo::cuda::kSlabHashSetKeysPerSlab + 1u;
  ManagedSetTable table(1, 0);
  auto *keys = managed_alloc<std::uint32_t>(kCount);
  auto *statuses = managed_alloc<std::uint32_t>(kCount);
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(statuses, nullptr);

  for (std::uint32_t i = 0; i < kCount; ++i)
    keys[i] = i + 1u;

  init_slab_hash_set(table.view());
  insert(table.view(), keys, kCount, statuses);

  std::uint32_t inserted = 0;
  std::uint32_t overflow = 0;
  for (std::uint32_t i = 0; i < kCount; ++i) {
    if (statuses[i] == algo::cuda::kSlabHashStatusInserted)
      ++inserted;
    if (statuses[i] == algo::cuda::kSlabHashStatusOverflow)
      ++overflow;
  }
  EXPECT_EQ(inserted, algo::cuda::kSlabHashSetKeysPerSlab);
  EXPECT_EQ(overflow, 1u);

  cudaFree(statuses);
  cudaFree(keys);
}

} // namespace
