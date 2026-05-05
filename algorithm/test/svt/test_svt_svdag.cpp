#include <algo/svt/svdag.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(SVDAG, DefaultsToEmpty) {
    algo::svt::SVDAG dag;

    EXPECT_FALSE(dag.get_voxel(0, 0, 0));
    EXPECT_FALSE(dag.get_voxel(17, 9, 31));
    EXPECT_EQ(dag.max_level(), algo::svt::DEFAULT_MAX_DEPTH);
    EXPECT_EQ(dag.max_depth(), dag.max_level());
    EXPECT_EQ(dag.world_voxel_count(), 1u << dag.max_level());
    EXPECT_EQ(dag.leaf_count(), 0u);
}

TEST(SVDAG, UsesConstructorMaxLevelForBounds) {
    algo::svt::SVDAGBuilder builder(3);

    EXPECT_NO_THROW(builder.set_voxel(7, 7, 7, true));
    EXPECT_THROW(builder.set_voxel(8, 0, 0, true), std::out_of_range);
    EXPECT_THROW(algo::svt::SVDAGBuilder(32), std::out_of_range);
    EXPECT_THROW(algo::svt::SVDAG(32), std::out_of_range);
}

TEST(SVDAG, MaxLevelZeroRepresentsSingleVoxel) {
    algo::svt::SVDAGBuilder builder(0);

    algo::svt::SVDAG empty = builder.build();
    EXPECT_EQ(empty.world_voxel_count(), 1u);
    EXPECT_EQ(empty.node_count(), 0u);
    EXPECT_FALSE(empty.get_voxel(0, 0, 0));
    EXPECT_THROW(static_cast<void>(empty.get_voxel(1, 0, 0)),
                 std::out_of_range);

    builder.set_voxel(0, 0, 0, true);
    algo::svt::SVDAG dag = builder.build();
    EXPECT_TRUE(dag.get_voxel(0, 0, 0));
    EXPECT_EQ(dag.node_count(), 0u);
}

TEST(SVDAG, BuilderSetsAndResetsVoxel) {
    algo::svt::SVDAGBuilder builder(4);

    builder.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(builder.build().get_voxel(5, 6, 7));

    builder.set_voxel(5, 6, 7, false);
    algo::svt::SVDAG dag = builder.build();
    EXPECT_FALSE(dag.get_voxel(5, 6, 7));
    EXPECT_EQ(dag.leaf_count(), 0u);
}

TEST(SVDAG, BuilderBuildPreservesVoxelData) {
    algo::svt::SVDAGBuilder builder(4);

    builder.set_voxel(1, 2, 3, true);
    builder.set_voxel(12, 3, 7, true);
    builder.set_voxel(1, 2, 3, false);

    algo::svt::SVDAG dag = builder.build();

    EXPECT_FALSE(dag.get_voxel(1, 2, 3));
    EXPECT_TRUE(dag.get_voxel(12, 3, 7));
    EXPECT_EQ(dag.node_count(), 4u);
}

TEST(SVDAG, CompactionMergesIdenticalSubtreesByLevel) {
    algo::svt::SVDAGBuilder builder(3);

    builder.set_voxel(0, 0, 0, true);
    builder.set_voxel(4, 0, 0, true);

    algo::svt::SVDAG dag = builder.build();

    EXPECT_TRUE(dag.get_voxel(0, 0, 0));
    EXPECT_TRUE(dag.get_voxel(4, 0, 0));
    EXPECT_FALSE(dag.get_voxel(0, 0, 1));
    EXPECT_EQ(dag.node_counts_by_level(),
              (std::vector<std::size_t>{1u, 1u, 1u}));
    EXPECT_EQ(dag.node_count(), 3u);
}

TEST(SVDAG, CompactionCollapsesUniformFilledAndEmptyNodes) {
    algo::svt::SVDAGBuilder builder(2);

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                builder.set_voxel(x, y, z, true);
            }
        }
    }

    algo::svt::SVDAG dag = builder.build();

    EXPECT_TRUE(dag.get_voxel(0, 0, 0));
    EXPECT_TRUE(dag.get_voxel(1, 1, 1));
    EXPECT_FALSE(dag.get_voxel(2, 0, 0));
    EXPECT_EQ(dag.node_counts_by_level(), (std::vector<std::size_t>{1u, 0u}));
    EXPECT_EQ(dag.node_count(), 1u);
}

TEST(SVDAG, ReportsNodeStorageBytesWithoutLeaves) {
    algo::svt::SVDAGBuilder builder(3);

    builder.set_voxel(0, 0, 0, true);
    algo::svt::SVDAG dag = builder.build();

    EXPECT_EQ(dag.leaf_count(), 0u);
    EXPECT_EQ(dag.node_count(), 3u);
    EXPECT_EQ(dag.node_storage_bytes(), 5u * sizeof(uint32_t));
    EXPECT_LT(dag.node_storage_bytes(),
              dag.node_count() * 8u * sizeof(uint32_t));
    EXPECT_GE(dag.memory_usage_bytes(), dag.node_storage_bytes());
}

TEST(SVDAG, PackedStorageUsesFilledMaskWithoutChildPointerPayload) {
    algo::svt::SVDAGBuilder builder(1);

    builder.set_voxel(0, 0, 0, true);
    builder.set_voxel(1, 1, 1, true);
    algo::svt::SVDAG dag = builder.build();

    EXPECT_TRUE(dag.get_voxel(0, 0, 0));
    EXPECT_TRUE(dag.get_voxel(1, 1, 1));
    EXPECT_FALSE(dag.get_voxel(1, 0, 0));
    EXPECT_EQ(dag.node_count(), 1u);
    EXPECT_EQ(dag.node_storage_bytes(), sizeof(uint32_t));
}

TEST(LabeledSVDAG, DefaultsToEmpty) {
    algo::svt::LabeledSVDAG dag;

    EXPECT_FALSE(dag.get_voxel(0, 0, 0));
    EXPECT_FALSE(dag.get_voxel(17, 9, 31));
    EXPECT_EQ(dag.max_level(), algo::svt::DEFAULT_MAX_DEPTH);
    EXPECT_EQ(dag.max_depth(), dag.max_level());
    EXPECT_EQ(dag.world_voxel_count(), 1u << dag.max_level());
    EXPECT_EQ(dag.leaf_count(), 0u);
}

TEST(LabeledSVDAG, BuildLabeledPreservesVoxelData) {
    algo::svt::SVDAGBuilder builder(4);

    builder.set_voxel(1, 2, 3, true);
    builder.set_voxel(12, 3, 7, true);
    builder.set_voxel(1, 2, 3, false);

    algo::svt::LabeledSVDAG dag = builder.build_labeled();

    EXPECT_FALSE(dag.get_voxel(1, 2, 3));
    EXPECT_TRUE(dag.get_voxel(12, 3, 7));
}

TEST(LabeledSVDAG, BuildLabeledCollapsesNodeWhoseChildrenShareSubtree) {
    algo::svt::SVDAGBuilder builder(2);

    for (uint32_t z = 0; z < 4; z += 2) {
        for (uint32_t y = 0; y < 4; y += 2) {
            for (uint32_t x = 0; x < 4; x += 2) {
                builder.set_voxel(x, y, z, true);
            }
        }
    }

    algo::svt::LabeledSVDAG dag = builder.build_labeled();

    EXPECT_TRUE(dag.get_voxel(0, 0, 0));
    EXPECT_TRUE(dag.get_voxel(2, 0, 0));
    EXPECT_TRUE(dag.get_voxel(2, 2, 2));
    EXPECT_FALSE(dag.get_voxel(1, 0, 0));
    EXPECT_FALSE(dag.get_voxel(3, 2, 2));
    EXPECT_EQ(dag.node_counts_by_level(), (std::vector<std::size_t>{0u, 1u}));
    EXPECT_EQ(dag.node_count(), 1u);
}
