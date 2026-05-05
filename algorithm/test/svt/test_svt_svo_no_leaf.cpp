#include <algo/svt/svo_no_leaf.hpp>
#include <gtest/gtest.h>

TEST(SvoNoLeaf, DefaultsToEmpty) {
    algo::svt::SVONoLeaf svo;
    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(17, 9, 31));
}

TEST(SvoNoLeaf, UsesConstructorMaxDepthForBounds) {
    algo::svt::SVONoLeaf svo(2);

    EXPECT_EQ(svo.max_depth(), 2u);
    EXPECT_EQ(svo.world_voxel_count(), 8u);
    EXPECT_NO_THROW(svo.set_voxel(7, 7, 7, true));
    EXPECT_THROW(svo.set_voxel(8, 0, 0, true), std::out_of_range);
}

TEST(SvoNoLeaf, SetAndResetVoxel) {
    algo::svt::SVONoLeaf svo;

    svo.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 0u);

    svo.set_voxel(5, 6, 7, false);
    EXPECT_FALSE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(SvoNoLeaf, CollapsesFullyFilledTerminalNodeBackIntoParentBit) {
    algo::svt::SVONoLeaf svo;

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x, y, z, true);
            }
        }
    }

    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(1, 1, 1));
    EXPECT_EQ(svo.leaf_count(), 0u);
    EXPECT_EQ(svo.node_count(), static_cast<std::size_t>(svo.max_depth()));
}

TEST(SvoNoLeaf, CollapsesIntermediateNodeWhenEightTerminalNodesBecomeFull) {
    algo::svt::SVONoLeaf svo;

    svo.set_voxel(0, 0, 0, true);
    const std::size_t node_count_before = svo.node_count();

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
    EXPECT_LT(svo.node_count(), node_count_before + 8u);
    EXPECT_EQ(svo.node_count(), static_cast<std::size_t>(svo.max_depth() - 1));
}

TEST(SvoNoLeaf, CollapsesFullyEmptiedTerminalNodeBackIntoParentBit) {
    algo::svt::SVONoLeaf svo;

    svo.set_voxel(0, 0, 0, true);
    svo.set_voxel(1, 0, 0, true);
    svo.set_voxel(0, 1, 0, true);

    const std::size_t node_count_before = svo.node_count();

    svo.set_voxel(0, 0, 0, false);
    svo.set_voxel(1, 0, 0, false);
    svo.set_voxel(0, 1, 0, false);

    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(1, 0, 0));
    EXPECT_FALSE(svo.get_voxel(0, 1, 0));
    EXPECT_EQ(svo.leaf_count(), 0u);
    EXPECT_LT(svo.node_count(), node_count_before);
}

TEST(SvoNoLeaf, CollapsesIntermediateNodeWhenEightTerminalNodesBecomeEmpty) {
    algo::svt::SVONoLeaf svo;

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
