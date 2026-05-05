#pragma once

template <class Config>
void apply_place_edits(TestGpuSvo &svo,
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

template <class Config>
void apply_destroy_edits(TestGpuSvo &svo,
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

    ASSERT_EQ((algo::svt::cuda::destroy_voxel_edits<Config>(
                  svo.view(), d_edits, static_cast<std::uint32_t>(edits.size()),
                  workspace, workspace_size)),
              cudaSuccess);
    sync_cuda();

    cudaFree(workspace);
    cudaFree(d_edits);
}

std::vector<std::uint8_t>
query_edit_voxels(TestGpuSvo &svo,
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

template <class Config> void expect_duplicate_place_edits_are_idempotent() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_place_edits<Config>(svo, {{0, 0, 0}});
    const auto counters_after_first = counters_for(svo);
    apply_place_edits<Config>(svo, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    const auto counters_after_duplicates = counters_for(svo);

    EXPECT_EQ(counters_after_duplicates.node_count,
              counters_after_first.node_count);
    EXPECT_EQ(counters_after_duplicates.leaf_count,
              counters_after_first.leaf_count);
    EXPECT_EQ(counters_after_duplicates.free_node_count,
              counters_after_first.free_node_count);
    EXPECT_EQ(counters_after_duplicates.free_leaf_count,
              counters_after_first.free_leaf_count);

    const auto results = query_edit_voxels(svo, {{0, 0, 0}, {1, 0, 0}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 0u);
}

template <class Config>
void expect_partial_destroy_preserves_leaf_and_other_bits() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_place_edits<Config>(svo, {{0, 0, 0}, {1, 0, 0}, {0, 0, 2}});
    apply_destroy_edits<Config>(svo, {{1, 0, 0}});

    const auto counters = counters_for(svo);
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 1u);

    const auto results =
        query_edit_voxels(svo, {{0, 0, 0}, {1, 0, 0}, {0, 0, 2}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 0u);
    EXPECT_EQ(results[2], 1u);

    const auto leaves =
        copy_from_device(svo.view().leaves, counters.leaf_count);
    ASSERT_EQ(leaves.size(), 1u);
    EXPECT_EQ(leaves[0].voxel_data_low, 1u);
    EXPECT_EQ(leaves[0].voxel_data_high, 1u);
}

template <class Config>
void expect_leaf_mask_low_high_boundary_bits_survive_edits() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_place_edits<Config>(svo, {{3, 3, 1}, {0, 0, 2}});
    auto results = query_edit_voxels(svo, {{3, 3, 1}, {0, 0, 2}, {2, 3, 1}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);

    const auto counters_after_place = counters_for(svo);
    auto leaves =
        copy_from_device(svo.view().leaves, counters_after_place.leaf_count);
    ASSERT_EQ(leaves.size(), 1u);
    EXPECT_EQ(leaves[0].voxel_data_low, 1u << 31u);
    EXPECT_EQ(leaves[0].voxel_data_high, 1u);

    apply_destroy_edits<Config>(svo, {{3, 3, 1}});
    results = query_edit_voxels(svo, {{3, 3, 1}, {0, 0, 2}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 0u);
    EXPECT_EQ(results[1], 1u);

    const auto counters_after_destroy = counters_for(svo);
    leaves =
        copy_from_device(svo.view().leaves, counters_after_destroy.leaf_count);
    ASSERT_EQ(leaves.size(), 1u);
    EXPECT_EQ(leaves[0].voxel_data_low, 0u);
    EXPECT_EQ(leaves[0].voxel_data_high, 1u);
}

template <class Config> void expect_out_of_bounds_edits_do_not_mutate_svo() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::kWorldVoxelCount;

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);
    const auto counters_after_reset = counters_for(svo);

    apply_place_edits<Config>(svo, {{kWorldVoxelCount, 0, 0},
                                    {0, kWorldVoxelCount, 0},
                                    {0, 0, kWorldVoxelCount}});
    auto counters = counters_for(svo);
    EXPECT_EQ(counters.node_count, counters_after_reset.node_count);
    EXPECT_EQ(counters.leaf_count, counters_after_reset.leaf_count);
    EXPECT_EQ(counters.free_node_count, counters_after_reset.free_node_count);
    EXPECT_EQ(counters.free_leaf_count, counters_after_reset.free_leaf_count);

    apply_place_edits<Config>(svo, {{0, 0, 0}, {kWorldVoxelCount, 0, 0}});
    counters = counters_for(svo);
    EXPECT_EQ(counters.node_count - counters.free_node_count,
              algo::svt::cuda::kMaxDepth);
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 1u);

    const auto results =
        query_edit_voxels(svo, {{0, 0, 0}, {kWorldVoxelCount, 0, 0}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 0u);
}

#define SVT_EDIT_INVARIANT_CONFIGS(X)                                          \
    X(PlainDepthwise,                                                          \
      algo::svt::cuda::EditConfig<algo::svt::cuda::PlainDepthwiseAllocation,   \
                                  algo::svt::cuda::VoxelCountDispatch>)

#define SVT_EDIT_INVARIANT_CHECKS(X, ConfigName, ...)                          \
    X(ConfigName, DuplicatePlaceEditsAreIdempotent,                            \
      expect_duplicate_place_edits_are_idempotent, __VA_ARGS__)                \
    X(ConfigName, PartialDestroyPreservesLeafAndOtherBits,                     \
      expect_partial_destroy_preserves_leaf_and_other_bits, __VA_ARGS__)       \
    X(ConfigName, LeafMaskLowHighBoundaryBitsSurviveEdits,                     \
      expect_leaf_mask_low_high_boundary_bits_survive_edits, __VA_ARGS__)      \
    X(ConfigName, OutOfBoundsEditsDoNotMutateSvo,                              \
      expect_out_of_bounds_edits_do_not_mutate_svo, __VA_ARGS__)

#define DEFINE_SVT_EDIT_INVARIANT_TEST(ConfigName, CheckName, CheckFn, ...)    \
    TEST(CudaSvtEditInvariant##CheckName, ConfigName) {                        \
        CheckFn<__VA_ARGS__>();                                                \
    }

#define DEFINE_SVT_EDIT_INVARIANT_TESTS_FOR_CONFIG(ConfigName, ...)            \
    SVT_EDIT_INVARIANT_CHECKS(DEFINE_SVT_EDIT_INVARIANT_TEST, ConfigName,      \
                              __VA_ARGS__)

SVT_EDIT_INVARIANT_CONFIGS(DEFINE_SVT_EDIT_INVARIANT_TESTS_FOR_CONFIG)

#undef DEFINE_SVT_EDIT_INVARIANT_TESTS_FOR_CONFIG
#undef DEFINE_SVT_EDIT_INVARIANT_TEST
#undef SVT_EDIT_INVARIANT_CHECKS
#undef SVT_EDIT_INVARIANT_CONFIGS
