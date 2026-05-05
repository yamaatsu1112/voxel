#pragma once

TEST(CudaSvt, UsesSvoIndividualLayoutConstants) {
    using namespace algo::svt::cuda;

    EXPECT_EQ(kBranchFactorExp, 1u);
    EXPECT_EQ(kGroupSize, 8u);
    EXPECT_EQ(kMaxDepth, 10u);
    EXPECT_EQ(kDefaultMaxNodeCount, 131072u);
    EXPECT_EQ(kDefaultMaxLeafCount, 524288u);
    EXPECT_EQ(TestGpuSvo::max_node_count, 256u);
    EXPECT_EQ(TestGpuSvo::max_leaf_count, 64u);
    EXPECT_EQ(sizeof(GpuSvoNode), sizeof(std::uint32_t) * 8u);
    EXPECT_EQ(
        (apply_voxel_edits_workspace_size<EditConfig<
             CompactAllDepthAllocation<Threadwise>, HostLeafCountDispatch>>(
            7u)),
        apply_voxel_edits_workspace_size(7u));
    EXPECT_GT(
        (apply_voxel_edits_workspace_size<
            EditConfig<CachedDepthwiseAllocation, VoxelCountDispatch>>(7u)),
        (apply_voxel_edits_workspace_size<
            EditConfig<PlainDepthwiseAllocation, VoxelCountDispatch>>(7u)));
    EXPECT_GT(
        (apply_voxel_edits_workspace_size<
            EditConfig<AllDepthAllocation, VoxelCountDispatch>>(7u)),
        (apply_voxel_edits_workspace_size<
            EditConfig<ScanDepthwiseAllocation, VoxelCountDispatch>>(7u)));
    EXPECT_LT(
        (apply_voxel_edits_workspace_size<
            EditConfig<CachedDepthwiseAllocation, VoxelCountDispatch>>(7u)),
        (apply_voxel_edits_workspace_size<
            EditConfig<AllDepthAllocation, VoxelCountDispatch>>(7u)));
    EXPECT_LT(
        (apply_voxel_edits_workspace_size<EditConfig<
             CompactAllDepthAllocation<Threadwise>, VoxelCountDispatch>>(7u)),
        (apply_voxel_edits_workspace_size<
            EditConfig<AllDepthAllocation, VoxelCountDispatch>>(7u)));
    EXPECT_EQ(
        (apply_voxel_edits_workspace_size<EditConfig<
             CompactAllDepthAllocation<Childwise>, VoxelCountDispatch>>(7u)),
        (apply_voxel_edits_workspace_size<EditConfig<
             CompactAllDepthAllocation<Threadwise>, VoxelCountDispatch>>(7u)));
    EXPECT_LT(
        (apply_voxel_edits_workspace_size<EditConfig<
             CompactAllDepthAllocation<Threadwise>, VoxelCountDispatch>>(7u)),
        (apply_voxel_edits_workspace_size<
            EditConfig<CompactAllDepthAllocation<
                           Threadwise, algo::svt::cuda::StoreStartDepth>,
                       VoxelCountDispatch>>(7u)));
    EXPECT_EQ(
        (apply_voxel_edits_workspace_size<EditConfig<
             CompactAllDepthAllocation<Childwise>, VoxelCountDispatch>>(7u)),
        (apply_voxel_edits_workspace_size<EditConfig<
             CompactAllDepthAllocation<Threadwise>, VoxelCountDispatch>>(7u)));
    EXPECT_EQ(
        (apply_voxel_edits_workspace_size<
            EditConfig<CompactAllDepthAllocation<Childwise, StoreStartDepth>,
                       VoxelCountDispatch>>(7u)),
        (apply_voxel_edits_workspace_size<
            EditConfig<CompactAllDepthAllocation<Threadwise, StoreStartDepth>,
                       VoxelCountDispatch>>(7u)));
}

template <class Config> void expect_place_query_results() {
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

    const auto workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(
            static_cast<std::uint32_t>(edits.size()));
    auto *workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);
    ASSERT_EQ((algo::svt::cuda::place_voxel_edits<Config>(
                  svo.view(), d_edits, static_cast<std::uint32_t>(edits.size()),
                  workspace, workspace_size)),
              cudaSuccess);
    sync_cuda();

    const std::vector<VoxelEdit> queries = {
        {0, 0, 0},          {1, 0, 0},
        {2, 0, 0},          {0, 0, 2},
        {0, 0, 3},          {4, 0, 0},
        {512, 512, 512},    {1023, 1023, 1023},
        {1022, 1023, 1023}, {algo::svt::cuda::kWorldVoxelCount, 0, 0},
    };
    auto *d_queries = device_alloc<VoxelEdit>(queries.size());
    auto *d_results = device_alloc<std::uint8_t>(queries.size());
    ASSERT_NE(d_queries, nullptr);
    ASSERT_NE(d_results, nullptr);
    copy_to_device(d_queries, queries);

    query_voxels_kernel<<<1, 64>>>(svo.view(), d_queries, d_results,
                                   static_cast<std::uint32_t>(queries.size()));
    sync_cuda();

    const auto results = copy_from_device(d_results, queries.size());
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);
    EXPECT_EQ(results[3], 1u);
    EXPECT_EQ(results[4], 0u);
    EXPECT_EQ(results[5], 1u);
    EXPECT_EQ(results[6], 1u);
    EXPECT_EQ(results[7], 1u);
    EXPECT_EQ(results[8], 0u);
    EXPECT_EQ(results[9], 0u);

    cudaFree(d_results);
    cudaFree(d_queries);
    cudaFree(workspace);
    cudaFree(d_edits);
}

#define SVT_PLACE_CONFIGS(X)                                                   \
    X(ScanDepthwise,                                                           \
      algo::svt::cuda::EditConfig<algo::svt::cuda::ScanDepthwiseAllocation,    \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(PlainDepthwise,                                                          \
      algo::svt::cuda::EditConfig<algo::svt::cuda::PlainDepthwiseAllocation,   \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(CachedDepthwise,                                                         \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CachedDepthwiseAllocation,  \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(CompactAllDepth,                                                         \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<  \
                                      algo::svt::cuda::Threadwise>,            \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(CompactAllDepthNodewiseOffsetSearch,                                     \
      algo::svt::cuda::EditConfig<                                             \
          algo::svt::cuda::CompactAllDepthAllocation<                          \
              algo::svt::cuda::Threadwise, algo::svt::cuda::StoreStartDepth,   \
              algo::svt::cuda::Nodewise<algo::svt::cuda::OffsetSearch>>,       \
          algo::svt::cuda::VoxelCountDispatch>)                                \
    X(CompactAllDepthRecoverStartDepthNodewiseOffsetSearch,                    \
      algo::svt::cuda::EditConfig<                                             \
          algo::svt::cuda::CompactAllDepthAllocation<                          \
              algo::svt::cuda::Threadwise, algo::svt::cuda::RecoverStartDepth, \
              algo::svt::cuda::Nodewise<algo::svt::cuda::OffsetSearch>>,       \
          algo::svt::cuda::VoxelCountDispatch>)                                \
    X(CompactAllDepthNodewiseExplicitRequests,                                 \
      algo::svt::cuda::EditConfig<                                             \
          algo::svt::cuda::CompactAllDepthAllocation<                          \
              algo::svt::cuda::Threadwise, algo::svt::cuda::StoreStartDepth,   \
              algo::svt::cuda::Nodewise<algo::svt::cuda::ExplicitRequests>>,   \
          algo::svt::cuda::VoxelCountDispatch>)                                \
    X(CompactAllDepthRecoverStartDepthNodewiseExplicitRequests,                \
      algo::svt::cuda::EditConfig<                                             \
          algo::svt::cuda::CompactAllDepthAllocation<                          \
              algo::svt::cuda::Threadwise, algo::svt::cuda::RecoverStartDepth, \
              algo::svt::cuda::Nodewise<algo::svt::cuda::ExplicitRequests>>,   \
          algo::svt::cuda::VoxelCountDispatch>)                                \
    X(CompactAllDepthChildwise,                                                \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<  \
                                      algo::svt::cuda::Childwise>,             \
                                  algo::svt::cuda::VoxelCountDispatch>)

#define SVT_PLACE_CHECKS(X, ConfigName, ...)                                   \
    X(ConfigName, QueryResults, expect_place_query_results, __VA_ARGS__)

#define DEFINE_SVT_PLACE_TEST(ConfigName, CheckName, CheckFn, ...)             \
    TEST(CudaSvtPlace##CheckName, ConfigName) { CheckFn<__VA_ARGS__>(); }

#define DEFINE_SVT_PLACE_TESTS_FOR_CONFIG(ConfigName, ...)                     \
    SVT_PLACE_CHECKS(DEFINE_SVT_PLACE_TEST, ConfigName, __VA_ARGS__)

SVT_PLACE_CONFIGS(DEFINE_SVT_PLACE_TESTS_FOR_CONFIG)

#undef DEFINE_SVT_PLACE_TESTS_FOR_CONFIG
#undef DEFINE_SVT_PLACE_TEST
#undef SVT_PLACE_CHECKS
#undef SVT_PLACE_CONFIGS
