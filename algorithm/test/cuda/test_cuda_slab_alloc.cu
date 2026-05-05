#include "cuda_test_utils.cuh"

#include <algo/cuda/hash_table/slab_alloc.cuh>
#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace {

using algo::cuda::SlabAlloc;

using cuda_test::copy_from_device;
using cuda_test::device_alloc;
using cuda_test::has_cuda_device;
using cuda_test::sync_cuda;

__global__ void initialize_slab_alloc_kernel(SlabAlloc::Device allocator) {
  SlabAlloc::initialize(allocator);
}

__global__ void allocate_warps_kernel(SlabAlloc::Device allocator,
                                      std::uint32_t *results,
                                      std::uint32_t allocation_count) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t global_warp = global_thread / 32u;
  if (global_warp >= allocation_count)
    return;

  const std::uint32_t allocated = SlabAlloc::allocate(allocator, lane);
  if (lane == 0)
    results[global_warp] = allocated;
}

__global__ void deallocate_kernel(SlabAlloc::Device allocator,
                                  std::uint32_t dynamic_slab_ordinal) {
  const std::uint32_t lane = threadIdx.x & 31u;
  SlabAlloc::deallocate(allocator, dynamic_slab_ordinal, lane);
}

__global__ void reuse_sequence_kernel(SlabAlloc::Device allocator,
                                      std::uint32_t *results) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t first = SlabAlloc::allocate(allocator, lane);
  SlabAlloc::deallocate(allocator, algo::cuda::kSlabHashNullSlab, lane);
  SlabAlloc::deallocate(allocator, allocator.slab_capacity, lane);
  const std::uint32_t second = SlabAlloc::allocate(allocator, lane);
  SlabAlloc::deallocate(allocator, first, lane);
  SlabAlloc::deallocate(allocator, first, lane);
  const std::uint32_t third = SlabAlloc::allocate(allocator, lane);

  if (lane == 0) {
    results[0] = first;
    results[1] = second;
    results[2] = third;
  }
}

void initialize_slab_alloc(SlabAlloc::Device allocator) {
  initialize_slab_alloc_kernel<<<256, 256>>>(allocator);
  sync_cuda();
}

std::vector<std::uint32_t> allocate_warps(SlabAlloc::Device allocator,
                                          std::uint32_t allocation_count) {
  std::uint32_t *results = device_alloc<std::uint32_t>(allocation_count);
  EXPECT_NE(results, nullptr);
  if (results == nullptr)
    return {};

  const std::uint32_t threads = 256;
  const std::uint32_t warps_per_block = threads / 32u;
  const std::uint32_t blocks =
      (allocation_count + warps_per_block - 1u) / warps_per_block;
  allocate_warps_kernel<<<blocks, threads>>>(allocator, results,
                                             allocation_count);
  sync_cuda();

  std::vector<std::uint32_t> host_results =
      copy_from_device(results, allocation_count);
  EXPECT_EQ(cudaFree(results), cudaSuccess);
  return host_results;
}

void fill_metadata(SlabAlloc::Host &host) {
  const std::uint32_t bitmap_word_count =
      SlabAlloc::Host::bitmap_word_count_for(host.slab_capacity);
  if (bitmap_word_count != 0) {
    ASSERT_EQ(cudaMemset(host.bitmap_words, 0xff,
                         sizeof(std::uint32_t) * bitmap_word_count),
              cudaSuccess);
  }

  ASSERT_EQ(cudaMemset(host.resident_change_attempts, 0xff,
                       sizeof(std::uint32_t) *
                           algo::cuda::kSlabAllocDefaultResidentWarpCount),
            cudaSuccess);
}

void expect_initialized(std::uint32_t slab_capacity) {
  SlabAlloc::Host host;
  ASSERT_NO_THROW(host.allocate(slab_capacity));

  const SlabAlloc::Device device = host.device_view();
  EXPECT_EQ(device.slab_capacity, slab_capacity);
  EXPECT_EQ(device.block_count,
            SlabAlloc::Host::block_count_for(slab_capacity));
  const std::uint32_t bitmap_word_count =
      SlabAlloc::Host::bitmap_word_count_for(slab_capacity);

  fill_metadata(host);
  initialize_slab_alloc(device);

  const std::vector<std::uint32_t> bitmap_words =
      copy_from_device(host.bitmap_words, bitmap_word_count);
  for (const std::uint32_t word : bitmap_words)
    EXPECT_EQ(word, 0u);

  const std::vector<std::uint32_t> resident_change_attempts =
      copy_from_device(host.resident_change_attempts,
                       algo::cuda::kSlabAllocDefaultResidentWarpCount);
  for (const std::uint32_t attempt : resident_change_attempts)
    EXPECT_EQ(attempt, 0u);
}

TEST(CudaSlabAlloc, InitializesMetadataForEmptyAndDynamicCapacities) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  expect_initialized(0);
  expect_initialized(17);
  expect_initialized(1031);
}

TEST(CudaSlabAlloc, HostMetadataCanMoveAndRelease) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  SlabAlloc::Host host;
  ASSERT_NO_THROW(host.allocate(2048));
  EXPECT_NE(host.bitmap_words, nullptr);
  EXPECT_NE(host.resident_change_attempts, nullptr);

  SlabAlloc::Host moved(std::move(host));
  EXPECT_EQ(host.bitmap_words, nullptr);
  EXPECT_EQ(host.resident_change_attempts, nullptr);
  EXPECT_EQ(moved.device_view().slab_capacity, 2048u);
  EXPECT_EQ(moved.device_view().block_count, 2u);

  moved.release();
  EXPECT_EQ(moved.bitmap_words, nullptr);
  EXPECT_EQ(moved.resident_change_attempts, nullptr);
  EXPECT_EQ(moved.device_view().slab_capacity, 0u);
  EXPECT_EQ(moved.device_view().block_count, 0u);
}

TEST(CudaSlabAlloc, AllocatesUniqueDynamicOrdinalsUntilExhausted) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCapacity = 65;
  SlabAlloc::Host host;
  ASSERT_NO_THROW(host.allocate(kCapacity));
  initialize_slab_alloc(host.device_view());

  std::vector<std::uint32_t> results =
      allocate_warps(host.device_view(), kCapacity + 17u);
  ASSERT_EQ(results.size(), kCapacity + 17u);

  std::vector<bool> seen(kCapacity, false);
  std::uint32_t allocated_count = 0;
  std::uint32_t failed_count = 0;
  for (const std::uint32_t result : results) {
    if (result == algo::cuda::kSlabHashNullSlab) {
      ++failed_count;
      continue;
    }

    ASSERT_LT(result, kCapacity);
    EXPECT_FALSE(seen[result]) << "duplicate ordinal " << result;
    seen[result] = true;
    ++allocated_count;
  }

  EXPECT_EQ(allocated_count, kCapacity);
  EXPECT_EQ(failed_count, 17u);
}

TEST(CudaSlabAlloc, ProbesBeyondInitialResidentBlockBeforeExhaustion) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  constexpr std::uint32_t kCapacity = algo::cuda::kSlabAllocBlockSlabs + 3u;
  SlabAlloc::Host host;
  ASSERT_NO_THROW(host.allocate(kCapacity));
  initialize_slab_alloc(host.device_view());

  std::vector<std::uint32_t> results =
      allocate_warps(host.device_view(), kCapacity);
  ASSERT_EQ(results.size(), kCapacity);

  std::vector<bool> seen(kCapacity, false);
  bool allocated_from_second_block = false;
  for (const std::uint32_t result : results) {
    ASSERT_NE(result, algo::cuda::kSlabHashNullSlab);
    ASSERT_LT(result, kCapacity);
    EXPECT_FALSE(seen[result]) << "duplicate ordinal " << result;
    seen[result] = true;
    allocated_from_second_block = allocated_from_second_block ||
                                  result >= algo::cuda::kSlabAllocBlockSlabs;
  }

  EXPECT_TRUE(allocated_from_second_block);
}

TEST(CudaSlabAlloc, DeallocationNoOpsAndDoubleDeallocationAllowReuse) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  SlabAlloc::Host host;
  ASSERT_NO_THROW(host.allocate(1));
  initialize_slab_alloc(host.device_view());

  std::uint32_t *results = device_alloc<std::uint32_t>(3);
  ASSERT_NE(results, nullptr);
  reuse_sequence_kernel<<<1, 32>>>(host.device_view(), results);
  sync_cuda();

  const std::vector<std::uint32_t> host_results = copy_from_device(results, 3);
  EXPECT_EQ(cudaFree(results), cudaSuccess);

  ASSERT_EQ(host_results.size(), 3u);
  EXPECT_EQ(host_results[0], 0u);
  EXPECT_EQ(host_results[1], algo::cuda::kSlabHashNullSlab);
  EXPECT_EQ(host_results[2], 0u);
}

TEST(CudaSlabAlloc, DeallocatingOutOfRangeOrdinalIsNoOp) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  SlabAlloc::Host host;
  ASSERT_NO_THROW(host.allocate(1));
  initialize_slab_alloc(host.device_view());

  std::vector<std::uint32_t> results = allocate_warps(host.device_view(), 1);
  ASSERT_EQ(results.size(), 1u);
  ASSERT_EQ(results[0], 0u);

  deallocate_kernel<<<1, 32>>>(host.device_view(), 99u);
  sync_cuda();

  results = allocate_warps(host.device_view(), 1);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0], algo::cuda::kSlabHashNullSlab);
}

} // namespace
