#pragma once

template <class Config>
void expect_frees_entire_materialized_leaf_subtree_overwritten_by_cell() {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_leaves<Config>(svo, {{0u, 1u, 0, 0, 0}},
                          algo::svt::cuda::EditOp::Place);

    const auto counters_before_overwrite = counters_for(svo);
    ASSERT_EQ(counters_before_overwrite.free_leaf_count, 0u);
    ASSERT_EQ(counters_before_overwrite.free_node_count, 0u);

    apply_terminal_nodes<Config>(svo, {{9u, 0u, 0, 0, 0}},
                         algo::svt::cuda::EditOp::Place);

    const auto counters_after_overwrite = counters_for(svo);
    EXPECT_EQ(counters_after_overwrite.free_leaf_count -
                  counters_before_overwrite.free_leaf_count,
              1u);
    EXPECT_EQ(counters_after_overwrite.free_node_count -
                  counters_before_overwrite.free_node_count,
              1u);

    const auto results = query_voxels(svo, {{0, 0, 0}, {7, 7, 7}, {8, 0, 0}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);
}

template <class Config>
void expect_frees_filled_intermediate_node_overwritten_by_parent_cell() {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_nodes<Config>(svo, {{9u, 0u, 0, 0, 0}},
                         algo::svt::cuda::EditOp::Place);
    const auto counters_after_child = counters_for(svo);

    apply_terminal_nodes<Config>(svo, {{8u, 0u, 0, 0, 0}},
                         algo::svt::cuda::EditOp::Place);
    const auto counters_after_parent = counters_for(svo);

    EXPECT_EQ(counters_after_parent.free_leaf_count -
                  counters_after_child.free_leaf_count,
              0u);
    EXPECT_EQ(counters_after_parent.free_node_count -
                  counters_after_child.free_node_count,
              1u);

    const auto results = query_voxels(svo, {{0, 0, 0}, {15, 15, 15}, {16, 0, 0}});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 0u);
}

template <class Config>
void expect_frees_multi_level_middle_subtree_overwritten_by_cell() {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    apply_terminal_leaves<Config>(
        svo,
        std::vector<algo::svt::cuda::TerminalLeafInput>{
            {static_cast<std::uint32_t>(leaf_key(0u, 0u, 0u)), 1u, 0, 0, 0},
            {static_cast<std::uint32_t>(leaf_key(16u, 0u, 0u)), 1u, 0, 0, 0},
        },
        algo::svt::cuda::EditOp::Place);

    const auto counters_before_overwrite = counters_for(svo);
    ASSERT_EQ(counters_before_overwrite.free_leaf_count, 0u);
    ASSERT_EQ(counters_before_overwrite.free_node_count, 0u);

    apply_terminal_nodes<Config>(svo, {{5u, 0u, 0, 0, 0}},
                         algo::svt::cuda::EditOp::Place);

    const auto counters_after_overwrite = counters_for(svo);
    EXPECT_EQ(counters_after_overwrite.free_leaf_count -
                  counters_before_overwrite.free_leaf_count,
              2u);
    EXPECT_EQ(counters_after_overwrite.free_node_count -
                  counters_before_overwrite.free_node_count,
              9u);

    const auto results =
        query_voxels(svo, {{0, 0, 0}, {64, 0, 0}, {127, 127, 127}, {128, 0, 0}});
    ASSERT_EQ(results.size(), 4u);
    EXPECT_EQ(results[0], 1u);
    EXPECT_EQ(results[1], 1u);
    EXPECT_EQ(results[2], 1u);
    EXPECT_EQ(results[3], 0u);
}

using TerminalDepthwiseReleaseConfig = algo::svt::cuda::TerminalEditConfig<
    algo::svt::cuda::TerminalCompactAllDepthAllocation,
    algo::svt::cuda::TerminalDepthwiseRelease>;
using TerminalFrontierReleaseConfig = algo::svt::cuda::TerminalEditConfig<
    algo::svt::cuda::TerminalCompactAllDepthAllocation,
    algo::svt::cuda::TerminalFrontierRelease>;

#define TERMINAL_EDIT_RELEASE_CONFIGS(X)                                       \
    X(Depthwise, TerminalDepthwiseReleaseConfig)                               \
    X(Frontier, TerminalFrontierReleaseConfig)

#define DEFINE_TERMINAL_EDIT_RELEASE_TESTS(Name, Config)                       \
    TEST(CudaTerminalEditRelease##Name,                                        \
         FreesEntireMaterializedLeafSubtreeOverwrittenByCell) {                \
        expect_frees_entire_materialized_leaf_subtree_overwritten_by_cell<      \
            Config>();                                                         \
    }                                                                          \
    TEST(CudaTerminalEditRelease##Name,                                        \
         FreesFilledIntermediateNodeOverwrittenByParentCell) {                 \
        expect_frees_filled_intermediate_node_overwritten_by_parent_cell<       \
            Config>();                                                         \
    }                                                                          \
    TEST(CudaTerminalEditRelease##Name,                                        \
         FreesMultiLevelMiddleSubtreeOverwrittenByCell) {                      \
        expect_frees_multi_level_middle_subtree_overwritten_by_cell<Config>();  \
    }

TERMINAL_EDIT_RELEASE_CONFIGS(DEFINE_TERMINAL_EDIT_RELEASE_TESTS)

#undef DEFINE_TERMINAL_EDIT_RELEASE_TESTS
#undef TERMINAL_EDIT_RELEASE_CONFIGS
