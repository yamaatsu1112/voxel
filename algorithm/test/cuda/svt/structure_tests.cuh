#pragma once

std::uint32_t expected_child_index(std::uint32_t x, std::uint32_t y,
                                   std::uint32_t z, std::uint32_t depth) {
    const std::uint32_t shift = (algo::svt::cuda::kMaxDepth - depth - 1u) *
                                    algo::svt::cuda::kBranchFactorExp +
                                algo::svt::cuda::kLeafVoxelCountExp;
    const std::uint32_t mask = (1u << algo::svt::cuda::kBranchFactorExp) - 1u;
    const std::uint32_t x_bits = (x >> shift) & mask;
    const std::uint32_t y_bits = (y >> shift) & mask;
    const std::uint32_t z_bits = (z >> shift) & mask;
    return x_bits | (y_bits << algo::svt::cuda::kBranchFactorExp) |
           (z_bits << (algo::svt::cuda::kBranchFactorExp * 2u));
}

bool host_node_has_child(const algo::svt::cuda::GpuSvoNode &node,
                         std::uint32_t child_index) {
    return (node.child_data[child_index] & algo::svt::cuda::kChildMaskBit) !=
           0u;
}

bool host_node_is_filled(const algo::svt::cuda::GpuSvoNode &node,
                         std::uint32_t child_index) {
    return (node.child_data[child_index] & algo::svt::cuda::kFilledBit) != 0u;
}

std::uint32_t host_node_child_index(const algo::svt::cuda::GpuSvoNode &node,
                                    std::uint32_t child_index) {
    return node.child_data[child_index] & algo::svt::cuda::kChildIndexMask;
}

template <class Config>
void place_structure_edits(
    TestGpuSvo &svo, const std::vector<algo::svt::cuda::VoxelEdit> &edits) {
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

void expect_materialized_path(
    const std::vector<algo::svt::cuda::GpuSvoNode> &nodes, std::uint32_t x,
    std::uint32_t y, std::uint32_t z, std::uint32_t *leaf_index_out) {
    std::uint32_t node_index = algo::svt::cuda::kRootNodeIndex;
    for (std::uint32_t depth = 0u; depth < algo::svt::cuda::kMaxDepth;
         ++depth) {
        ASSERT_LT(node_index, nodes.size());
        const std::uint32_t child_index = expected_child_index(x, y, z, depth);
        const auto &node = nodes[node_index];
        EXPECT_TRUE(host_node_has_child(node, child_index));
        EXPECT_FALSE(host_node_is_filled(node, child_index));
        node_index = host_node_child_index(node, child_index);
    }
    *leaf_index_out = node_index;
}

template <class Config>
void expect_same_leaf_edits_share_path_and_merge_leaf_mask() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::VoxelEdit;

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);
    place_structure_edits<Config>(svo, {{0, 0, 0}, {1, 0, 0}, {0, 0, 2}});

    const auto counters = counters_for(svo);
    EXPECT_EQ(counters.node_count - counters.free_node_count,
              algo::svt::cuda::kMaxDepth);
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 1u);

    const auto nodes = copy_from_device(svo.view().nodes, counters.node_count);
    const auto leaves =
        copy_from_device(svo.view().leaves, counters.leaf_count);
    std::uint32_t leaf_index = 0u;
    expect_materialized_path(nodes, 0u, 0u, 0u, &leaf_index);
    ASSERT_LT(leaf_index, leaves.size());
    EXPECT_EQ(leaves[leaf_index].voxel_data_low, 0b11u);
    EXPECT_EQ(leaves[leaf_index].voxel_data_high, 1u);
}

template <class Config>
void expect_distant_edits_materialize_independent_root_paths() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::VoxelEdit;

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);
    place_structure_edits<Config>(svo, {{0, 0, 0}, {2048, 0, 0}});

    const auto counters = counters_for(svo);
    EXPECT_EQ(counters.node_count - counters.free_node_count,
              1u + 2u * (algo::svt::cuda::kMaxDepth - 1u));
    EXPECT_EQ(counters.leaf_count - counters.free_leaf_count, 2u);

    const auto nodes = copy_from_device(svo.view().nodes, counters.node_count);
    const auto leaves =
        copy_from_device(svo.view().leaves, counters.leaf_count);
    ASSERT_FALSE(nodes.empty());

    const auto &root = nodes[algo::svt::cuda::kRootNodeIndex];
    EXPECT_TRUE(host_node_has_child(root, 0u));
    EXPECT_TRUE(host_node_has_child(root, 1u));
    for (std::uint32_t child = 2u; child < algo::svt::cuda::kGroupSize; ++child)
        EXPECT_FALSE(host_node_has_child(root, child));

    std::uint32_t first_leaf = 0u;
    std::uint32_t second_leaf = 0u;
    expect_materialized_path(nodes, 0u, 0u, 0u, &first_leaf);
    expect_materialized_path(nodes, 2048u, 0u, 0u, &second_leaf);
    ASSERT_LT(first_leaf, leaves.size());
    ASSERT_LT(second_leaf, leaves.size());
    EXPECT_NE(first_leaf, second_leaf);
    EXPECT_EQ(leaves[first_leaf].voxel_data_low, 1u);
    EXPECT_EQ(leaves[first_leaf].voxel_data_high, 0u);
    EXPECT_EQ(leaves[second_leaf].voxel_data_low, 1u);
    EXPECT_EQ(leaves[second_leaf].voxel_data_high, 0u);
}

#define SVT_STRUCTURE_CONFIGS(X)                                               \
    X(PlainDepthwise,                                                          \
      algo::svt::cuda::EditConfig<algo::svt::cuda::PlainDepthwiseAllocation,   \
                                  algo::svt::cuda::VoxelCountDispatch>)

#define SVT_STRUCTURE_CHECKS(X, ConfigName, ...)                               \
    X(ConfigName, SameLeafEditsSharePathAndMergeLeafMask,                      \
      expect_same_leaf_edits_share_path_and_merge_leaf_mask, __VA_ARGS__)      \
    X(ConfigName, DistantEditsMaterializeIndependentRootPaths,                 \
      expect_distant_edits_materialize_independent_root_paths, __VA_ARGS__)

#define DEFINE_SVT_STRUCTURE_TEST(ConfigName, CheckName, CheckFn, ...)         \
    TEST(CudaSvtStructure##CheckName, ConfigName) { CheckFn<__VA_ARGS__>(); }

#define DEFINE_SVT_STRUCTURE_TESTS_FOR_CONFIG(ConfigName, ...)                 \
    SVT_STRUCTURE_CHECKS(DEFINE_SVT_STRUCTURE_TEST, ConfigName, __VA_ARGS__)

SVT_STRUCTURE_CONFIGS(DEFINE_SVT_STRUCTURE_TESTS_FOR_CONFIG)

#undef DEFINE_SVT_STRUCTURE_TESTS_FOR_CONFIG
#undef DEFINE_SVT_STRUCTURE_TEST
#undef SVT_STRUCTURE_CHECKS
#undef SVT_STRUCTURE_CONFIGS
