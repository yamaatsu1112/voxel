#pragma once

template <class Config>
void place_edits(TestGpuSvo &svo,
                 const std::vector<algo::svt::cuda::VoxelEdit> &edits) {
    using algo::svt::cuda::VoxelEdit;

    auto *d_edits = device_alloc<VoxelEdit>(edits.size());
    ASSERT_NE(d_edits, nullptr);
    copy_to_device(d_edits, edits);

    const auto workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(
            static_cast<std::uint32_t>(edits.size()));
    auto *workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    ASSERT_EQ((algo::svt::cuda::place_voxel_edits<Config>(
                  svo.view(), d_edits, static_cast<std::uint32_t>(edits.size()),
                  workspace, workspace_size)),
              cudaSuccess);
    sync_cuda();

    cudaFree(workspace);
    cudaFree(d_edits);
}

std::vector<algo::svt::cuda::VoxelEdit>
voxel_box_edits(std::uint32_t size, algo::svt::cuda::VoxelEdit omitted) {
    using algo::svt::cuda::VoxelEdit;

    std::vector<VoxelEdit> edits;
    edits.reserve(size * size * size - 1u);
    for (std::uint32_t z = 0; z < size; ++z) {
        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                if (x == omitted.x && y == omitted.y && z == omitted.z)
                    continue;
                edits.push_back({x, y, z});
            }
        }
    }
    return edits;
}

std::vector<std::uint8_t>
query_voxels(TestGpuSvo &svo,
             const std::vector<algo::svt::cuda::VoxelEdit> &queries) {
    using algo::svt::cuda::VoxelEdit;

    auto *d_queries = device_alloc<VoxelEdit>(queries.size());
    auto *d_results = device_alloc<std::uint8_t>(queries.size());
    EXPECT_NE(d_queries, nullptr);
    EXPECT_NE(d_results, nullptr);
    copy_to_device(d_queries, queries);

    query_voxels_kernel<<<1, 64>>>(svo.view(), d_queries, d_results,
                                   static_cast<std::uint32_t>(queries.size()));
    sync_cuda();

    std::vector<std::uint8_t> results =
        copy_from_device(d_results, queries.size());
    cudaFree(d_results);
    cudaFree(d_queries);
    return results;
}

template <class Config>
void expect_collapses_materialized_leaf_when_final_voxel_is_placed() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::VoxelEdit;

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    const VoxelEdit final_voxel{3, 3, 3};
    place_edits<Config>(svo, voxel_box_edits(4u, final_voxel));

    auto counters = counters_for(svo);
    ASSERT_EQ(counters.leaf_count - counters.free_leaf_count, 1u);

    auto results = query_voxels(svo, {{0, 0, 0}, final_voxel});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 0u);

    place_edits<Config>(svo, {final_voxel});

    counters = counters_for(svo);
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 0u);
    EXPECT_EQ(counters.node_count - counters.free_node_count,
              algo::svt::cuda::kMaxDepth);

    results = query_voxels(svo, {{0, 0, 0}, final_voxel, {4, 0, 0}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);
}

template <class Config>
void expect_cascades_to_parent_when_final_child_becomes_full() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::VoxelEdit;

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    const VoxelEdit final_voxel{7, 7, 7};
    place_edits<Config>(svo, voxel_box_edits(8u, final_voxel));

    auto counters = counters_for(svo);
    ASSERT_EQ(counters.leaf_count - counters.free_leaf_count, 1u);
    ASSERT_EQ(counters.node_count - counters.free_node_count,
              algo::svt::cuda::kMaxDepth);

    auto results = query_voxels(svo, {{0, 0, 0}, final_voxel});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 0u);

    place_edits<Config>(svo, {final_voxel});

    counters = counters_for(svo);
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 0u);
    EXPECT_EQ(counters.node_count - counters.free_node_count,
              algo::svt::cuda::kMaxDepth - 1u);

    results = query_voxels(svo, {{0, 0, 0}, final_voxel, {8, 0, 0}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);
}

template <class Config>
void expect_cascades_across_multiple_intermediate_levels() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::VoxelEdit;

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    const VoxelEdit final_voxel{15, 15, 15};
    place_edits<Config>(svo, voxel_box_edits(16u, final_voxel));

    auto counters = counters_for(svo);
    ASSERT_EQ(counters.leaf_count - counters.free_leaf_count, 1u);
    ASSERT_EQ(counters.node_count - counters.free_node_count,
              algo::svt::cuda::kMaxDepth);

    auto results = query_voxels(svo, {{0, 0, 0}, final_voxel});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 0u);

    place_edits<Config>(svo, {final_voxel});

    counters = counters_for(svo);
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 0u);
    EXPECT_EQ(counters.node_count - counters.free_node_count,
              algo::svt::cuda::kMaxDepth - 2u);

    results = query_voxels(svo, {{0, 0, 0}, final_voxel, {16, 0, 0}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);
}

#define SVT_COLLAPSE_CONFIGS(X)                                                \
    X(PlainDepthwise,                                                          \
      algo::svt::cuda::EditConfig<algo::svt::cuda::PlainDepthwiseAllocation,   \
                                  algo::svt::cuda::VoxelCountDispatch>)

#define SVT_COLLAPSE_CHECKS(X, ConfigName, ...)                                \
    X(ConfigName, CollapsesMaterializedLeafWhenFinalVoxelIsPlaced,             \
      expect_collapses_materialized_leaf_when_final_voxel_is_placed,           \
      __VA_ARGS__)                                                             \
    X(ConfigName, CascadesToParentWhenFinalChildBecomesFull,                   \
      expect_cascades_to_parent_when_final_child_becomes_full, __VA_ARGS__)    \
    X(ConfigName, CascadesAcrossMultipleIntermediateLevels,                    \
      expect_cascades_across_multiple_intermediate_levels, __VA_ARGS__)

#define DEFINE_SVT_COLLAPSE_TEST(ConfigName, CheckName, CheckFn, ...)          \
    TEST(CudaSvtCollapse##CheckName, ConfigName) { CheckFn<__VA_ARGS__>(); }

#define DEFINE_SVT_COLLAPSE_TESTS_FOR_CONFIG(ConfigName, ...)                  \
    SVT_COLLAPSE_CHECKS(DEFINE_SVT_COLLAPSE_TEST, ConfigName, __VA_ARGS__)

SVT_COLLAPSE_CONFIGS(DEFINE_SVT_COLLAPSE_TESTS_FOR_CONFIG)

#undef DEFINE_SVT_COLLAPSE_TESTS_FOR_CONFIG
#undef DEFINE_SVT_COLLAPSE_TEST
#undef SVT_COLLAPSE_CHECKS
#undef SVT_COLLAPSE_CONFIGS
