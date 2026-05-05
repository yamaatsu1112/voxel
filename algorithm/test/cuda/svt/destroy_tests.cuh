#pragma once

template <class Config> void expect_destroy_from_collapsed_filled_leaf() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::VoxelEdit;

    std::vector<VoxelEdit> edits;
    edits.reserve(4u * 4u * 4u);
    for (std::uint32_t z = 0; z < 4; ++z) {
        for (std::uint32_t y = 0; y < 4; ++y) {
            for (std::uint32_t x = 0; x < 4; ++x)
                edits.push_back({x, y, z});
        }
    }

    auto *d_edits = device_alloc<VoxelEdit>(edits.size());
    ASSERT_NE(d_edits, nullptr);
    copy_to_device(d_edits, edits);

    const auto build_workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(
            static_cast<std::uint32_t>(edits.size()));
    auto *build_workspace = device_alloc<std::byte>(build_workspace_size);
    ASSERT_NE(build_workspace, nullptr);

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);
    ASSERT_EQ((algo::svt::cuda::place_voxel_edits<Config>(
                  svo.view(), d_edits, static_cast<std::uint32_t>(edits.size()),
                  build_workspace, build_workspace_size)),
              cudaSuccess);
    sync_cuda();
    const auto counters_after_build = counters_for(svo);
    ASSERT_EQ(counters_after_build.leaf_count -
                  counters_after_build.free_leaf_count,
              0u);

    const std::vector<VoxelEdit> destroyed = {{0, 0, 0}};
    auto *d_destroyed = device_alloc<VoxelEdit>(destroyed.size());
    ASSERT_NE(d_destroyed, nullptr);
    copy_to_device(d_destroyed, destroyed);

    const auto destroy_workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(
            static_cast<std::uint32_t>(destroyed.size()));
    auto *destroy_workspace = device_alloc<std::byte>(destroy_workspace_size);
    ASSERT_NE(destroy_workspace, nullptr);

    ASSERT_EQ((algo::svt::cuda::destroy_voxel_edits<Config>(
                  svo.view(), d_destroyed,
                  static_cast<std::uint32_t>(destroyed.size()),
                  destroy_workspace, destroy_workspace_size)),
              cudaSuccess);
    sync_cuda();

    const auto counters_after_destroy = counters_for(svo);
    EXPECT_EQ(counters_after_destroy.leaf_count -
                  counters_after_destroy.free_leaf_count,
              1u);

    const std::vector<VoxelEdit> queries = {{0, 0, 0}, {1, 0, 0}, {3, 3, 3}};
    auto *d_queries = device_alloc<VoxelEdit>(queries.size());
    auto *d_results = device_alloc<std::uint8_t>(queries.size());
    ASSERT_NE(d_queries, nullptr);
    ASSERT_NE(d_results, nullptr);
    copy_to_device(d_queries, queries);

    query_voxels_kernel<<<1, 64>>>(svo.view(), d_queries, d_results,
                                   static_cast<std::uint32_t>(queries.size()));
    sync_cuda();

    const auto results = copy_from_device(d_results, queries.size());
    EXPECT_EQ(results[0], 0u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 1u);

    cudaFree(d_results);
    cudaFree(d_queries);
    cudaFree(destroy_workspace);
    cudaFree(d_destroyed);
    cudaFree(build_workspace);
    cudaFree(d_edits);
}

template <class Config> void expect_destroy_reuses_freed_storage() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::VoxelEdit;

    const std::vector<VoxelEdit> edits = {{0, 0, 0},       {1, 0, 0},
                                          {0, 0, 2},       {4, 0, 0},
                                          {512, 512, 512}, {1023, 1023, 1023}};

    auto *d_edits = device_alloc<VoxelEdit>(edits.size());
    ASSERT_NE(d_edits, nullptr);
    copy_to_device(d_edits, edits);

    const auto build_workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(
            static_cast<std::uint32_t>(edits.size()));
    auto *build_workspace = device_alloc<std::byte>(build_workspace_size);
    ASSERT_NE(build_workspace, nullptr);

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);
    ASSERT_EQ((algo::svt::cuda::place_voxel_edits<Config>(
                  svo.view(), d_edits, static_cast<std::uint32_t>(edits.size()),
                  build_workspace, build_workspace_size)),
              cudaSuccess);
    sync_cuda();

    const std::vector<VoxelEdit> destroyed = {{512, 512, 512},
                                              {1023, 1023, 1023}};
    auto *d_destroyed = device_alloc<VoxelEdit>(destroyed.size());
    ASSERT_NE(d_destroyed, nullptr);
    copy_to_device(d_destroyed, destroyed);

    const auto destroy_workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(
            static_cast<std::uint32_t>(destroyed.size()));
    auto *destroy_workspace = device_alloc<std::byte>(destroy_workspace_size);
    ASSERT_NE(destroy_workspace, nullptr);

    ASSERT_EQ((algo::svt::cuda::destroy_voxel_edits<Config>(
                  svo.view(), d_destroyed,
                  static_cast<std::uint32_t>(destroyed.size()),
                  destroy_workspace, destroy_workspace_size)),
              cudaSuccess);
    sync_cuda();

    const auto counters_after_destroy = counters_for(svo);
    const auto free_leaf_count_after_destroy =
        counters_after_destroy.free_leaf_count;
    const auto free_node_count_after_destroy =
        counters_after_destroy.free_node_count;
    EXPECT_GT(free_leaf_count_after_destroy, 0u);
    EXPECT_GT(free_node_count_after_destroy, 0u);

    const std::vector<VoxelEdit> queries_after_destroy = {
        {0, 0, 0}, {512, 512, 512}, {1023, 1023, 1023}};
    auto *d_queries = device_alloc<VoxelEdit>(queries_after_destroy.size());
    auto *d_results = device_alloc<std::uint8_t>(queries_after_destroy.size());
    ASSERT_NE(d_queries, nullptr);
    ASSERT_NE(d_results, nullptr);
    copy_to_device(d_queries, queries_after_destroy);

    query_voxels_kernel<<<1, 64>>>(
        svo.view(), d_queries, d_results,
        static_cast<std::uint32_t>(queries_after_destroy.size()));
    sync_cuda();

    auto results = copy_from_device(d_results, queries_after_destroy.size());
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 0u);
    EXPECT_EQ(results[2], 0u);

    const std::vector<VoxelEdit> place_again = {{1023, 1023, 1023}};
    auto *d_place_again = device_alloc<VoxelEdit>(place_again.size());
    ASSERT_NE(d_place_again, nullptr);
    copy_to_device(d_place_again, place_again);

    const auto place_workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(
            static_cast<std::uint32_t>(place_again.size()));
    auto *place_workspace = device_alloc<std::byte>(place_workspace_size);
    ASSERT_NE(place_workspace, nullptr);

    ASSERT_EQ((algo::svt::cuda::place_voxel_edits<Config>(
                  svo.view(), d_place_again,
                  static_cast<std::uint32_t>(place_again.size()),
                  place_workspace, place_workspace_size)),
              cudaSuccess);
    sync_cuda();

    const auto counters_after_place = counters_for(svo);
    EXPECT_LT(counters_after_place.free_leaf_count,
              free_leaf_count_after_destroy);
    EXPECT_LT(counters_after_place.free_node_count,
              free_node_count_after_destroy);

    query_voxels_kernel<<<1, 64>>>(svo.view(), d_place_again, d_results, 1u);
    sync_cuda();
    results = copy_from_device(d_results, 1u);
    EXPECT_EQ(results[0], 1u);

    cudaFree(place_workspace);
    cudaFree(d_place_again);
    cudaFree(d_results);
    cudaFree(d_queries);
    cudaFree(destroy_workspace);
    cudaFree(d_destroyed);
    cudaFree(build_workspace);
    cudaFree(d_edits);
}

#define SVT_DESTROY_CONFIGS(X)                                                 \
    X(Default, algo::svt::cuda::detail::DefaultEditConfig)                     \
    X(CachedDepthwise,                                                         \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CachedDepthwiseAllocation,  \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(PlainDepthwise,                                                          \
      algo::svt::cuda::EditConfig<algo::svt::cuda::PlainDepthwiseAllocation,   \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(AllDepth,                                                                \
      algo::svt::cuda::EditConfig<algo::svt::cuda::AllDepthAllocation,         \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(CompactAllDepth,                                                         \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<  \
                                      algo::svt::cuda::Threadwise>,            \
                                  algo::svt::cuda::VoxelCountDispatch>)

#define SVT_DESTROY_CHECKS(X, ConfigName, ...)                                 \
    X(ConfigName, FromCollapsedFilledLeaf,                                     \
      expect_destroy_from_collapsed_filled_leaf, __VA_ARGS__)                  \
    X(ConfigName, ReusesFreedStorage, expect_destroy_reuses_freed_storage,     \
      __VA_ARGS__)

#define DEFINE_SVT_DESTROY_TEST(ConfigName, CheckName, CheckFn, ...)           \
    TEST(CudaSvtDestroy##CheckName, ConfigName) { CheckFn<__VA_ARGS__>(); }

#define DEFINE_SVT_DESTROY_TESTS_FOR_CONFIG(ConfigName, ...)                   \
    SVT_DESTROY_CHECKS(DEFINE_SVT_DESTROY_TEST, ConfigName, __VA_ARGS__)

SVT_DESTROY_CONFIGS(DEFINE_SVT_DESTROY_TESTS_FOR_CONFIG)

#undef DEFINE_SVT_DESTROY_TESTS_FOR_CONFIG
#undef DEFINE_SVT_DESTROY_TEST
#undef SVT_DESTROY_CHECKS
#undef SVT_DESTROY_CONFIGS
