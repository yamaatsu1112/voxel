#pragma once

TEST(CudaTerminalEditDestroy, PlacesAndDestroysFullTerminalCell) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    const std::vector<algo::svt::cuda::TerminalNodeInput> node = {
        {10u, 0u, 0, 0, 0}};
    apply_terminal_nodes(svo, node, algo::svt::cuda::EditOp::Place);

    auto results = query_voxels(svo, {{0, 0, 0}, {3, 3, 3}, {4, 0, 0}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);

    apply_terminal_nodes(svo, node, algo::svt::cuda::EditOp::Destroy);

    results = query_voxels(svo, {{0, 0, 0}, {3, 3, 3}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 0u);
    EXPECT_EQ(results[1], 0u);
}

TEST(CudaTerminalEditDestroy, HandlesAncestorAndDescendantBatch) {
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

    auto status = algo::svt::cuda::detail::apply_terminal_edits<
        algo::svt::cuda::DefaultTerminalEditConfig::allocation,
        algo::svt::cuda::DefaultTerminalEditConfig::release>(
        svo.view(), d_root, 1u, d_leaf, 1u, algo::svt::cuda::EditOp::Place,
        workspace, workspace_size);
    ASSERT_EQ(status, cudaSuccess);
    sync_cuda();

    status = algo::svt::cuda::detail::apply_terminal_edits<
        algo::svt::cuda::DefaultTerminalEditConfig::allocation,
        algo::svt::cuda::DefaultTerminalEditConfig::release>(
        svo.view(), d_root, 1u, d_leaf, 1u, algo::svt::cuda::EditOp::Destroy,
        workspace, workspace_size);
    ASSERT_EQ(status, cudaSuccess);
    sync_cuda();

    const auto results = query_voxels(svo, {{0, 0, 0}, {4095, 4095, 4095}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 0u);
    EXPECT_EQ(results[1], 0u);

    cudaFree(workspace);
    cudaFree(d_leaf);
    cudaFree(d_root);
}

TEST(CudaTerminalEditDestroy, HandlesAncestorAndDescendantCells) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    const std::vector<algo::svt::cuda::TerminalNodeInput> nodes = {
        {0u, 0u, 0, 0, 0},
        {9u, 0u, 0, 0, 0},
    };
    apply_terminal_nodes(svo, nodes, algo::svt::cuda::EditOp::Place);
    apply_terminal_nodes(svo, nodes, algo::svt::cuda::EditOp::Destroy);

    const auto results = query_voxels(svo, {{0, 0, 0}, {4095, 4095, 4095}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 0u);
    EXPECT_EQ(results[1], 0u);
}
