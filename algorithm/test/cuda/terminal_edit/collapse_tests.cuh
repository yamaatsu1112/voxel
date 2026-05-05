#pragma once

TEST(CudaTerminalEditCollapse, CollapsesFullTerminalLeaf) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_nodes(svo, {{10u, 0u, 0, 0, 0}},
                         algo::svt::cuda::EditOp::Place);

    const auto results = query_voxels(svo, {{0, 0, 0}, {3, 3, 3}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);

    const auto counters = counters_for(svo);
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 0u);
}

TEST(CudaTerminalEditCollapse, AllocatesLeafWhenEditingUniformTerminalLeaf) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_nodes(svo, {{10u, 0u, 0, 0, 0}},
                         algo::svt::cuda::EditOp::Place);
    const auto counters_after_collapse = counters_for(svo);
    ASSERT_EQ(counters_after_collapse.free_leaf_count, 0u);

    apply_terminal_leaves(svo, {{0u, 1u, 0, 0, 0}},
                          algo::svt::cuda::EditOp::Destroy);

    const auto counters_after_reuse = counters_for(svo);
    EXPECT_EQ(counters_after_reuse.leaf_count,
              counters_after_collapse.leaf_count + 1u);
    EXPECT_EQ(counters_after_reuse.free_leaf_count, 0u);

    const auto results = query_voxels(svo, {{0, 0, 0}, {1, 0, 0}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], 0u);
    EXPECT_EQ(results[1], 1u);
}
