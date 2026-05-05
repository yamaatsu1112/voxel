#pragma once

TEST(CudaTerminalEditPlace, PlacesTerminalLeafIntoWorldSvt) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_leaves(svo, {{0u, 0b11u, 0, 0, 0}},
                          algo::svt::cuda::EditOp::Place);

    const auto results =
        query_voxels(svo, {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);
}

TEST(CudaTerminalEditPlace, PlacesRootWithoutLeafExpansion) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_nodes(svo, {{0u, 0u, 0, 0, 0}},
                         algo::svt::cuda::EditOp::Place);

    const auto results = query_voxels(
        svo, {{0, 0, 0}, {2048, 2048, 2048}, {4095, 4095, 4095}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 1u);
}

TEST(CudaTerminalEditPlace, HandlesAncestorAndDescendantBatch) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    const algo::svt::cuda::TerminalNodeInput root{0u, 0u, 0, 0, 0};
    const algo::svt::cuda::TerminalLeafInput leaf{0u, 1u, 0, 0, 0};
    auto* d_root = device_alloc<algo::svt::cuda::TerminalNodeInput>(1u);
    auto* d_leaf = device_alloc<algo::svt::cuda::TerminalLeafInput>(1u);
    ASSERT_NE(d_root, nullptr);
    ASSERT_NE(d_leaf, nullptr);
    copy_to_device(d_root, std::vector<algo::svt::cuda::TerminalNodeInput>{root});
    copy_to_device(d_leaf, std::vector<algo::svt::cuda::TerminalLeafInput>{leaf});
    const auto workspace_size =
        algo::svt::cuda::detail::apply_terminal_edits_workspace_size<
            algo::svt::cuda::DefaultTerminalEditConfig::allocation,
            algo::svt::cuda::DefaultTerminalEditConfig::release>(1u, 1u,
                                                                    16u);
    auto* workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    const auto status = algo::svt::cuda::detail::apply_terminal_edits<
        algo::svt::cuda::DefaultTerminalEditConfig::allocation,
        algo::svt::cuda::DefaultTerminalEditConfig::release>(
        svo.view(), d_root, 1u, d_leaf, 1u, algo::svt::cuda::EditOp::Place,
        workspace, workspace_size);
    ASSERT_EQ(status, cudaSuccess);
    sync_cuda();

    const auto results = query_voxels(svo, {{0, 0, 0}, {4095, 4095, 4095}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);

    cudaFree(workspace);
    cudaFree(d_leaf);
    cudaFree(d_root);
}

TEST(CudaTerminalEditPlace, HandlesAncestorAndDescendantCells) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_nodes(svo, {{0u, 0u, 0, 0, 0}, {9u, 0u, 0, 0, 0}},
                         algo::svt::cuda::EditOp::Place);

    const auto results = query_voxels(svo, {{0, 0, 0}, {4095, 4095, 4095}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
}

TEST(CudaTerminalEditPlace, LeavesExpectedSvoShapeAfterLeafAllocation) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_leaves(svo, {{0u, 1u, 0, 0, 0}},
                          algo::svt::cuda::EditOp::Place);

    const auto counters = counters_for(svo);
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 1u);
    EXPECT_EQ(counters.node_count - counters.free_node_count,
              algo::svt::cuda::kMaxDepth);
}
