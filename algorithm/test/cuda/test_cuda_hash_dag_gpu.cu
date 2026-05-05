#include "cuda_test_utils.cuh"

#include <algo/svt/hash_dag_gpu.cuh>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace {

using cuda_test::copy_from_device;
using cuda_test::copy_scalar_from_device;
using cuda_test::copy_to_device;
using cuda_test::device_alloc;
using cuda_test::has_cuda_device;
using cuda_test::sync_cuda;

using TestDag = algo::svt::hash_dag_gpu::HashDagGpu<64, 64>;

__global__ void
query_voxels_kernel(algo::svt::hash_dag_gpu::DeviceHashDagGpu dag,
                    const algo::svt::hash_dag_gpu::VoxelEdit *queries,
                    std::uint8_t *results, std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const auto query = queries[index];
  results[index] =
      algo::svt::hash_dag_gpu::get_voxel(dag, query.x, query.y, query.z) ? 1u
                                                                         : 0u;
}

__global__ void
ref_at_depth_kernel(algo::svt::hash_dag_gpu::DeviceHashDagGpu dag,
                    std::uint32_t x, std::uint32_t y, std::uint32_t z,
                    std::uint32_t depth, std::uint32_t *result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;

  std::uint32_t ref = *dag.root_ref;
  for (std::uint32_t level = 0u; level < depth; ++level) {
    const std::uint32_t child =
        algo::svt::hash_dag_gpu::detail::child_index_for_voxel_level(x, y, z,
                                                                     level);
    ref = algo::svt::hash_dag_gpu::detail::child_ref(dag, ref, child);
  }
  *result = ref;
}

std::uint32_t ref_at_depth(algo::svt::hash_dag_gpu::DeviceHashDagGpu dag,
                           std::uint32_t x, std::uint32_t y, std::uint32_t z,
                           std::uint32_t depth) {
  auto *d_result = device_alloc<std::uint32_t>(1);
  EXPECT_NE(d_result, nullptr);

  ref_at_depth_kernel<<<1, 1>>>(dag, x, y, z, depth, d_result);
  sync_cuda();

  const std::uint32_t result = copy_scalar_from_device(d_result);
  cudaFree(d_result);
  return result;
}

std::vector<std::uint8_t>
query_voxels(algo::svt::hash_dag_gpu::DeviceHashDagGpu dag,
             const std::vector<algo::svt::hash_dag_gpu::VoxelEdit> &queries) {
  if (queries.empty())
    return {};

  auto *d_queries =
      device_alloc<algo::svt::hash_dag_gpu::VoxelEdit>(queries.size());
  auto *d_results = device_alloc<std::uint8_t>(queries.size());
  EXPECT_NE(d_queries, nullptr);
  EXPECT_NE(d_results, nullptr);
  copy_to_device(d_queries, queries);

  const auto count = static_cast<std::uint32_t>(queries.size());
  const std::uint32_t block_size = 64u;
  const std::uint32_t blocks = (count + block_size - 1u) / block_size;
  query_voxels_kernel<<<blocks, block_size>>>(dag, d_queries, d_results, count);
  sync_cuda();

  auto results = copy_from_device(d_results, queries.size());
  cudaFree(d_results);
  cudaFree(d_queries);
  return results;
}

cudaError_t place_voxel_edits(
    algo::svt::hash_dag_gpu::DeviceHashDagGpu dag,
    const std::vector<algo::svt::hash_dag_gpu::VoxelEdit> &edits) {
  auto *d_edits =
      device_alloc<algo::svt::hash_dag_gpu::VoxelEdit>(edits.size());
  EXPECT_NE(d_edits, nullptr);
  copy_to_device(d_edits, edits);

  const auto workspace_size =
      algo::svt::hash_dag_gpu::apply_voxel_edits_workspace_size(
          static_cast<std::uint32_t>(edits.size()));
  auto *workspace = device_alloc<std::byte>(workspace_size);
  EXPECT_NE(workspace, nullptr);

  const cudaError_t status = algo::svt::hash_dag_gpu::place_voxel_edits(
      dag, d_edits, static_cast<std::uint32_t>(edits.size()), workspace,
      workspace_size);

  cudaFree(workspace);
  cudaFree(d_edits);
  return status;
}

cudaError_t destroy_voxel_edits(
    algo::svt::hash_dag_gpu::DeviceHashDagGpu dag,
    const std::vector<algo::svt::hash_dag_gpu::VoxelEdit> &edits) {
  auto *d_edits =
      device_alloc<algo::svt::hash_dag_gpu::VoxelEdit>(edits.size());
  EXPECT_NE(d_edits, nullptr);
  copy_to_device(d_edits, edits);

  const auto workspace_size =
      algo::svt::hash_dag_gpu::apply_voxel_edits_workspace_size(
          static_cast<std::uint32_t>(edits.size()));
  auto *workspace = device_alloc<std::byte>(workspace_size);
  EXPECT_NE(workspace, nullptr);

  const cudaError_t status = algo::svt::hash_dag_gpu::destroy_voxel_edits(
      dag, d_edits, static_cast<std::uint32_t>(edits.size()), workspace,
      workspace_size);

  cudaFree(workspace);
  cudaFree(d_edits);
  return status;
}

std::uint32_t voxel_key(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return x | (y << 10u) | (z << 20u);
}

std::vector<algo::svt::hash_dag_gpu::VoxelEdit>
make_spread_edits(std::uint32_t count) {
  std::vector<algo::svt::hash_dag_gpu::VoxelEdit> edits;
  edits.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (i % 19u == 0u) {
      edits.push_back({1024u + i, 0u, 0u});
      continue;
    }

    edits.push_back(
        {(i * 37u) & 1023u, (i * 91u + 3u) & 1023u, (i * 53u + 7u) & 1023u});
  }
  return edits;
}

TEST(CudaHashDagGpu, EncodesDagRefsWithFourBitItemSize) {
  using namespace algo::svt::hash_dag_gpu;

  const std::uint32_t ref = detail::make_table_ref(9u, 0x01234567u);
  EXPECT_EQ(detail::ref_item_words(ref), 9u);
  EXPECT_EQ(detail::ref_hash_ptr(ref), 0x01234567u);
  EXPECT_TRUE(detail::is_table_ref(ref));
  EXPECT_TRUE(detail::is_empty_ref(kEmptyRef));
  EXPECT_TRUE(detail::is_full_ref(kFullRef));
}

TEST(CudaHashDagGpu, BuildsLeafKeysAtFourVoxelGranularity) {
  using namespace algo::svt::hash_dag_gpu;

  EXPECT_EQ(kInnerDepth, 8u);
  EXPECT_EQ(detail::make_leaf_key(0u, 0u, 0u),
            detail::make_leaf_key(3u, 3u, 3u));
  EXPECT_NE(detail::make_leaf_key(3u, 0u, 0u),
            detail::make_leaf_key(4u, 0u, 0u));
  EXPECT_EQ(detail::leaf_local_bit_index(0u, 0u, 0u), 0u);
  EXPECT_EQ(detail::leaf_local_bit_index(3u, 3u, 3u), 63u);
}

TEST(CudaHashDagGpu, PlacesAndDestroysSingleVoxels) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using algo::svt::hash_dag_gpu::VoxelEdit;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  ASSERT_EQ(
      place_voxel_edits(dag.view(), {{0, 0, 0}, {1, 1, 1}, {1023, 1023, 1023}}),
      cudaSuccess);

  const std::vector<VoxelEdit> queries = {
      {0, 0, 0},          {1, 1, 1},          {1, 0, 0},    {2, 0, 0},
      {1023, 1023, 1023}, {1022, 1023, 1023}, {1024, 0, 0},
  };
  auto results = query_voxels(dag.view(), queries);
  ASSERT_EQ(results.size(), queries.size());
  EXPECT_EQ(results[0], 1u);
  EXPECT_EQ(results[1], 1u);
  EXPECT_EQ(results[2], 0u);
  EXPECT_EQ(results[3], 0u);
  EXPECT_EQ(results[4], 1u);
  EXPECT_EQ(results[5], 0u);
  EXPECT_EQ(results[6], 0u);

  ASSERT_EQ(destroy_voxel_edits(dag.view(), {{0, 0, 0}}), cudaSuccess);

  results = query_voxels(dag.view(), queries);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(results[1], 1u);
  EXPECT_EQ(results[4], 1u);
}

TEST(CudaHashDagGpu, EmptyBatchIsNoOp) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using algo::svt::hash_dag_gpu::VoxelEdit;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  ASSERT_EQ(algo::svt::hash_dag_gpu::place_voxel_edits(dag.view(), nullptr, 0u,
                                                       nullptr, 0u),
            cudaSuccess);

  const std::vector<VoxelEdit> queries = {{0, 0, 0}, {1023, 1023, 1023}};
  const auto results = query_voxels(dag.view(), queries);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(results[1], 0u);
}

TEST(CudaHashDagGpu, IgnoresOutOfBoundsEdits) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using algo::svt::hash_dag_gpu::VoxelEdit;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  ASSERT_EQ(place_voxel_edits(dag.view(), {{1024, 0, 0}}), cudaSuccess);

  const std::vector<VoxelEdit> queries = {{0, 0, 0}, {1023, 1023, 1023}};
  const auto results = query_voxels(dag.view(), queries);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(results[1], 0u);
}

TEST(CudaHashDagGpu, EditingOneDeduplicatedPatternDoesNotMutateAnother) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using algo::svt::hash_dag_gpu::VoxelEdit;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  ASSERT_EQ(place_voxel_edits(dag.view(), {{0, 0, 0},
                                           {1, 0, 0},
                                           {0, 1, 0},
                                           {1, 1, 0},
                                           {16, 0, 0},
                                           {17, 0, 0},
                                           {16, 1, 0},
                                           {17, 1, 0}}),
            cudaSuccess);

  ASSERT_EQ(destroy_voxel_edits(dag.view(), {{0, 0, 0}}), cudaSuccess);

  const std::vector<VoxelEdit> queries = {
      {0, 0, 0}, {1, 0, 0}, {16, 0, 0}, {17, 0, 0}, {16, 1, 0}};
  const auto results = query_voxels(dag.view(), queries);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(results[1], 1u);
  EXPECT_EQ(results[2], 1u);
  EXPECT_EQ(results[3], 1u);
  EXPECT_EQ(results[4], 1u);
}

TEST(CudaHashDagGpu, BatchEditsAreIdempotentWithinFinalNode) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using algo::svt::hash_dag_gpu::VoxelEdit;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  ASSERT_EQ(
      place_voxel_edits(
          dag.view(), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}}),
      cudaSuccess);

  const std::vector<VoxelEdit> queries = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                                          {1, 1, 1}, {0, 0, 1}, {2, 0, 0}};
  const auto results = query_voxels(dag.view(), queries);
  EXPECT_EQ(results[0], 1u);
  EXPECT_EQ(results[1], 1u);
  EXPECT_EQ(results[2], 1u);
  EXPECT_EQ(results[3], 1u);
  EXPECT_EQ(results[4], 0u);
  EXPECT_EQ(results[5], 0u);
}

TEST(CudaHashDagGpu, FillsAndPartiallyDestroysFinalNode) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using algo::svt::hash_dag_gpu::VoxelEdit;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  std::vector<VoxelEdit> edits;
  for (std::uint32_t z = 0; z < 2; ++z) {
    for (std::uint32_t y = 0; y < 2; ++y) {
      for (std::uint32_t x = 0; x < 2; ++x)
        edits.push_back({x, y, z});
    }
  }
  ASSERT_EQ(place_voxel_edits(dag.view(), edits), cudaSuccess);

  ASSERT_EQ(destroy_voxel_edits(dag.view(), {{0, 0, 0}, {1, 1, 1}}),
            cudaSuccess);

  const std::vector<VoxelEdit> queries = {
      {0, 0, 0}, {1, 1, 1}, {1, 0, 0}, {0, 1, 1}, {2, 0, 0}};
  const auto results = query_voxels(dag.view(), queries);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(results[1], 0u);
  EXPECT_EQ(results[2], 1u);
  EXPECT_EQ(results[3], 1u);
  EXPECT_EQ(results[4], 0u);
}

TEST(CudaHashDagGpu, UpdatesSiblingFinalNodesUnderSameParent) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using algo::svt::hash_dag_gpu::VoxelEdit;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  ASSERT_EQ(place_voxel_edits(dag.view(), {{0, 0, 0}, {2, 0, 0}}), cudaSuccess);
  ASSERT_EQ(destroy_voxel_edits(dag.view(), {{0, 0, 0}}), cudaSuccess);

  const std::vector<VoxelEdit> queries = {
      {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
  const auto results = query_voxels(dag.view(), queries);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(results[1], 0u);
  EXPECT_EQ(results[2], 1u);
  EXPECT_EQ(results[3], 0u);
}

TEST(CudaHashDagGpu, EightPartialSiblingFinalNodesCreateNineWordParent) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using namespace algo::svt::hash_dag_gpu;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  std::vector<VoxelEdit> edits;
  std::vector<VoxelEdit> queries;
  std::vector<std::uint8_t> expected;
  for (std::uint32_t z = 0; z < 2; ++z) {
    for (std::uint32_t y = 0; y < 2; ++y) {
      for (std::uint32_t x = 0; x < 2; ++x) {
        const VoxelEdit placed{x * 4u, y * 4u, z * 4u};
        const VoxelEdit empty{x * 4u + 1u, y * 4u, z * 4u};
        edits.push_back(placed);
        queries.push_back(placed);
        expected.push_back(1u);
        queries.push_back(empty);
        expected.push_back(0u);
      }
    }
  }

  ASSERT_EQ(place_voxel_edits(dag.view(), edits), cudaSuccess);

  const auto results = query_voxels(dag.view(), queries);
  ASSERT_EQ(results.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(results[i], expected[i]) << "query index " << i;

  const std::uint32_t parent_ref =
      ref_at_depth(dag.view(), 0u, 0u, 0u, kInnerDepth - 1u);
  ASSERT_TRUE(detail::is_table_ref(parent_ref));
  EXPECT_EQ(detail::ref_item_words(parent_ref), 9u);
}

TEST(CudaHashDagGpu, BoundaryBatchSizesMatchCpuReference) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using algo::svt::hash_dag_gpu::VoxelEdit;
  constexpr std::uint32_t counts[] = {31u, 32u, 33u, 255u, 256u, 257u};

  for (const std::uint32_t count : counts) {
    TestDag dag;
    ASSERT_EQ(dag.status(), cudaSuccess);

    const auto edits = make_spread_edits(count);
    std::unordered_set<std::uint32_t> occupied;
    for (const auto &edit : edits) {
      if (edit.x < 1024u && edit.y < 1024u && edit.z < 1024u)
        occupied.insert(voxel_key(edit.x, edit.y, edit.z));
    }

    ASSERT_EQ(place_voxel_edits(dag.view(), edits), cudaSuccess)
        << "count " << count;

    std::vector<VoxelEdit> queries = edits;
    for (std::uint32_t i = 0; i < 32u; ++i) {
      queries.push_back({(i * 13u + 5u) & 1023u, (i * 17u + 11u) & 1023u,
                         (i * 29u + 23u) & 1023u});
    }
    queries.push_back({1024u, 0u, 0u});

    const auto results = query_voxels(dag.view(), queries);
    ASSERT_EQ(results.size(), queries.size()) << "count " << count;

    for (std::size_t i = 0; i < queries.size(); ++i) {
      const auto query = queries[i];
      const bool valid = query.x < 1024u && query.y < 1024u && query.z < 1024u;
      const std::uint8_t expected =
          valid && occupied.count(voxel_key(query.x, query.y, query.z)) != 0u
              ? 1u
              : 0u;
      EXPECT_EQ(results[i], expected)
          << "count " << count << ", query index " << i;
    }
  }
}

TEST(CudaHashDagGpu, RejectsInvalidApiArgumentsAndKeepsExistingDag) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using namespace algo::svt::hash_dag_gpu;

  TestDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);
  ASSERT_EQ(place_voxel_edits(dag.view(), {{0, 0, 0}}), cudaSuccess);

  const std::vector<VoxelEdit> edits = {{2, 0, 0}};
  auto *d_edits = device_alloc<VoxelEdit>(edits.size());
  ASSERT_NE(d_edits, nullptr);
  copy_to_device(d_edits, edits);

  const auto workspace_size = apply_voxel_edits_workspace_size(
      static_cast<std::uint32_t>(edits.size()));
  auto *workspace = device_alloc<std::byte>(workspace_size);
  ASSERT_NE(workspace, nullptr);

  EXPECT_EQ(
      place_voxel_edits(dag.view(), nullptr, 1u, workspace, workspace_size),
      cudaErrorInvalidValue);
  EXPECT_EQ(place_voxel_edits(DeviceHashDagGpu{}, d_edits, 1u, workspace,
                              workspace_size),
            cudaErrorInvalidValue);
  EXPECT_EQ(place_voxel_edits(dag.view(), d_edits, 1u, nullptr, workspace_size),
            cudaErrorInvalidValue);
  EXPECT_EQ(place_voxel_edits(dag.view(), d_edits, 1u, workspace,
                              workspace_size - 1u),
            cudaErrorInvalidValue);

  cudaFree(workspace);
  cudaFree(d_edits);

  const std::vector<VoxelEdit> queries = {{0, 0, 0}, {2, 0, 0}};
  const auto results = query_voxels(dag.view(), queries);
  EXPECT_EQ(results[0], 1u);
  EXPECT_EQ(results[1], 0u);
}

TEST(CudaHashDagGpu, HashOverflowDoesNotPublishPartialRoot) {
  cudaError_t status = cudaSuccess;
  if (!has_cuda_device(&status)) {
    GTEST_SKIP() << "CUDA device is not available: "
                 << cudaGetErrorString(status);
  }

  using TinyDag = algo::svt::hash_dag_gpu::HashDagGpu<1, 0>;
  using algo::svt::hash_dag_gpu::VoxelEdit;

  TinyDag dag;
  ASSERT_EQ(dag.status(), cudaSuccess);

  const auto edits = make_spread_edits(512u);
  ASSERT_EQ(place_voxel_edits(dag.view(), edits), cudaErrorMemoryAllocation);

  const std::vector<VoxelEdit> queries = {
      {37, 94, 60}, {74, 185, 113}, {111, 276, 166}, {148, 367, 219}};
  const auto results = query_voxels(dag.view(), queries);
  for (const auto result : results)
    EXPECT_EQ(result, 0u);
}

} // namespace
