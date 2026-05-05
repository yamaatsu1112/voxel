#pragma once

template <class Config> void expect_places_disjoint_leaf_paths_in_one_batch() {
  expect_cuda_device_or_skip();

  TestGpuSvo svo;
  ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

  apply_terminal_leaves<Config>(
      svo,
      {
          {static_cast<std::uint32_t>(leaf_key(0u, 0u, 0u)), 0b0001u, 0, 0, 0},
          {static_cast<std::uint32_t>(leaf_key(512u, 0u, 0u)), 0b0010u, 0, 0,
           0},
      },
      algo::svt::cuda::EditOp::Place);

  const auto results =
      query_voxels(svo, {{0, 0, 0}, {1, 0, 0}, {2049, 0, 0}, {2048, 0, 0}});
  ASSERT_EQ(results.size(), 4u);
  EXPECT_EQ(results[0], 1u);
  EXPECT_EQ(results[1], 0u);
  EXPECT_EQ(results[2], 1u);
  EXPECT_EQ(results[3], 0u);
}

template <class Config>
void expect_refines_existing_terminal_node_with_leaf_edit() {
  expect_cuda_device_or_skip();

  TestGpuSvo svo;
  ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

  apply_terminal_nodes<Config>(svo, {{9u, 0u, 0, 0, 0}},
                               algo::svt::cuda::EditOp::Place);
  apply_terminal_leaves<Config>(svo, {{0u, 0b0001u, 0, 0, 0}},
                                algo::svt::cuda::EditOp::Destroy);

  const auto results =
      query_voxels(svo, {{0, 0, 0}, {1, 0, 0}, {7, 7, 7}, {8, 0, 0}});
  ASSERT_EQ(results.size(), 4u);
  EXPECT_EQ(results[0], 0u);
  EXPECT_EQ(results[1], 1u);
  EXPECT_EQ(results[2], 1u);
  EXPECT_EQ(results[3], 0u);
}

template <class Config>
void expect_places_three_leaf_siblings_sharing_new_parent() {
  expect_cuda_device_or_skip();

  TestGpuSvo svo;
  ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

  apply_terminal_leaves<Config>(
      svo,
      {
          {static_cast<std::uint32_t>(leaf_key(0u, 0u, 0u)), 0b0001u, 0, 0, 0},
          {static_cast<std::uint32_t>(leaf_key(1u, 0u, 0u)), 0b0010u, 0, 0, 0},
          {static_cast<std::uint32_t>(leaf_key(2u, 0u, 0u)), 0b0100u, 0, 0, 0},
      },
      algo::svt::cuda::EditOp::Place);

  const auto results =
      query_voxels(svo, {{0, 0, 0}, {5, 0, 0}, {10, 0, 0}, {11, 0, 0}});
  ASSERT_EQ(results.size(), 4u);
  EXPECT_EQ(results[0], 1u);
  EXPECT_EQ(results[1], 1u);
  EXPECT_EQ(results[2], 1u);
  EXPECT_EQ(results[3], 0u);
}

#define TERMINAL_EDIT_ALLOCATION_CONFIGS(X)                                    \
  X(PlainDepthwise, algo::svt::cuda::TerminalEditConfig<                       \
                        algo::svt::cuda::TerminalPlainDepthwiseAllocation,     \
                        algo::svt::cuda::TerminalDepthwiseRelease>)            \
  X(CompactAllDepth, algo::svt::cuda::TerminalEditConfig<                      \
                         algo::svt::cuda::TerminalCompactAllDepthAllocation,   \
                         algo::svt::cuda::TerminalDepthwiseRelease>)

#define TERMINAL_EDIT_ALLOCATION_CHECKS(X, ConfigName, ...)                    \
  X(ConfigName, PlacesDisjointLeafPathsInOneBatch,                             \
    expect_places_disjoint_leaf_paths_in_one_batch, __VA_ARGS__)               \
  X(ConfigName, RefinesExistingTerminalNodeWithLeafEdit,                       \
    expect_refines_existing_terminal_node_with_leaf_edit, __VA_ARGS__)         \
  X(ConfigName, PlacesThreeLeafSiblingsSharingNewParent,                       \
    expect_places_three_leaf_siblings_sharing_new_parent, __VA_ARGS__)

#define DEFINE_TERMINAL_EDIT_ALLOCATION_TEST(ConfigName, CheckName, CheckFn,   \
                                             ...)                              \
  TEST(CudaTerminalEditAllocation##CheckName, ConfigName) {                    \
    CheckFn<__VA_ARGS__>();                                                    \
  }

#define DEFINE_TERMINAL_EDIT_ALLOCATION_TESTS_FOR_CONFIG(ConfigName, ...)      \
  TERMINAL_EDIT_ALLOCATION_CHECKS(DEFINE_TERMINAL_EDIT_ALLOCATION_TEST,        \
                                  ConfigName, __VA_ARGS__)

TERMINAL_EDIT_ALLOCATION_CONFIGS(
    DEFINE_TERMINAL_EDIT_ALLOCATION_TESTS_FOR_CONFIG)

#undef DEFINE_TERMINAL_EDIT_ALLOCATION_TESTS_FOR_CONFIG
#undef DEFINE_TERMINAL_EDIT_ALLOCATION_TEST
#undef TERMINAL_EDIT_ALLOCATION_CHECKS
#undef TERMINAL_EDIT_ALLOCATION_CONFIGS
