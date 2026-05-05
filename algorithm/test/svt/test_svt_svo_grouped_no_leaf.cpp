#include <algo/svt/svo_grouped_no_leaf.hpp>
#include <gtest/gtest.h>

TEST(SvoGroupedNoLeaf, DefaultsToEmpty) {
    algo::svt::SVOGroupedNoLeaf svo;
    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(17, 9, 31));
}

TEST(SvoGroupedNoLeaf, UsesConstructorMaxDepthForBounds) {
    algo::svt::SVOGroupedNoLeaf svo(2);

    EXPECT_EQ(svo.max_depth(), 2u);
    EXPECT_EQ(svo.world_voxel_count(), 8u);
    EXPECT_NO_THROW(svo.set_voxel(7, 7, 7, true));
    EXPECT_THROW(svo.set_voxel(8, 0, 0, true), std::out_of_range);
}

TEST(SvoGroupedNoLeaf, AllocatesTerminalNodesInEightNodeBlocks) {
    algo::svt::SVOGroupedNoLeaf svo(1);

    svo.set_voxel(0, 0, 0, true);
    EXPECT_EQ(svo.node_count(), 9u);

    svo.set_voxel(2, 0, 0, true);
    EXPECT_EQ(svo.node_count(), 9u);
    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(2, 0, 0));
}

TEST(SvoGroupedNoLeaf, SetAndResetVoxelReclaimsTerminalBlock) {
    algo::svt::SVOGroupedNoLeaf svo;

    svo.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(svo.get_voxel(5, 6, 7));
    EXPECT_GT(svo.node_count(), 1u);

    svo.set_voxel(5, 6, 7, false);
    EXPECT_FALSE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.node_count(), 1u);
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(SvoGroupedNoLeaf, ReclaimsTerminalBlockWhenParentRegionBecomesFull) {
    algo::svt::SVOGroupedNoLeaf svo;

    svo.set_voxel(0, 0, 0, true);
    const std::size_t node_count_with_partial_terminal = svo.node_count();

    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                svo.set_voxel(x, y, z, true);
            }
        }
    }

    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(3, 3, 3));
    EXPECT_EQ(svo.leaf_count(), 0u);
    EXPECT_LT(svo.node_count(), node_count_with_partial_terminal);
}

TEST(SvoGroupedNoLeaf,
     CollapsesIntermediateNodeWhenEightTerminalNodesBecomeEmpty) {
    algo::svt::SVOGroupedNoLeaf svo;

    constexpr uint32_t L = 2;

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * L, y * L, z * L, true);
            }
        }
    }

    const std::size_t node_count_with_terminals = svo.node_count();

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * L, y * L, z * L, false);
            }
        }
    }

    EXPECT_EQ(svo.leaf_count(), 0u);
    EXPECT_LT(svo.node_count(), node_count_with_terminals);
    EXPECT_EQ(svo.node_count(), 1u);
}
