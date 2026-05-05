#include <algo/svt/svo64_grouped_no_leaf.hpp>
#include <gtest/gtest.h>

TEST(Svo64GroupedNoLeaf, DefaultsToEmpty) {
    algo::svt::SVO64GroupedNoLeaf svo;
    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(17, 9, 31));
}

TEST(Svo64GroupedNoLeaf, UsesConstructorMaxDepthForBounds) {
    algo::svt::SVO64GroupedNoLeaf svo(2);

    EXPECT_EQ(svo.max_depth(), 2u);
    EXPECT_EQ(svo.world_voxel_count(), algo::svt::LEAF_VOXEL_COUNT << 4);
    EXPECT_NO_THROW(svo.set_voxel(63, 63, 63, true));
    EXPECT_THROW(svo.set_voxel(64, 0, 0, true), std::out_of_range);
}

TEST(Svo64GroupedNoLeaf, AllocatesLeafNodesOnlyForOccupiedLeaves) {
    algo::svt::SVO64GroupedNoLeaf svo(1);

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    svo.set_voxel(0, 0, 0, true);
    EXPECT_EQ(svo.node_count(), 2u);

    svo.set_voxel(3 * L, 0, 0, true);
    EXPECT_EQ(svo.node_count(), 3u);
    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(3 * L, 0, 0));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(Svo64GroupedNoLeaf, SetAndResetVoxelReclaimsEmptyLeafNode) {
    algo::svt::SVO64GroupedNoLeaf svo(1);

    svo.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.node_count(), 2u);

    svo.set_voxel(5, 6, 7, false);
    EXPECT_FALSE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.node_count(), 1u);
}

TEST(Svo64GroupedNoLeaf, DoesNotCollapseFullyFilledLeafNode) {
    algo::svt::SVO64GroupedNoLeaf svo(1);

    for (uint32_t z = 0; z < algo::svt::LEAF_VOXEL_COUNT; ++z) {
        for (uint32_t y = 0; y < algo::svt::LEAF_VOXEL_COUNT; ++y) {
            for (uint32_t x = 0; x < algo::svt::LEAF_VOXEL_COUNT; ++x) {
                svo.set_voxel(x, y, z, true);
            }
        }
    }

    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(3, 3, 3));
    EXPECT_EQ(svo.node_count(), 2u);
}

TEST(Svo64GroupedNoLeaf,
     CollapsesIntermediateNodeWhenSixtyFourLeavesBecomeEmpty) {
    algo::svt::SVO64GroupedNoLeaf svo(2);

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                svo.set_voxel(x * L, y * L, z * L, true);
            }
        }
    }

    const std::size_t node_count_with_leaves = svo.node_count();
    EXPECT_EQ(node_count_with_leaves, 66u);

    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                svo.set_voxel(x * L, y * L, z * L, false);
            }
        }
    }

    EXPECT_LT(svo.node_count(), node_count_with_leaves);
    EXPECT_EQ(svo.node_count(), 1u);
}

TEST(Svo64GroupedNoLeaf, PreservesMixedParentWhenSiblingLeafBecomesEmpty) {
    algo::svt::SVO64GroupedNoLeaf svo(2);

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    for (uint32_t z = 0; z < L; ++z) {
        for (uint32_t y = 0; y < L; ++y) {
            for (uint32_t x = 0; x < L; ++x) {
                svo.set_voxel(L + x, y, z, true);
            }
        }
    }

    svo.set_voxel(0, 0, 0, true);
    EXPECT_EQ(svo.node_count(), 4u);

    svo.set_voxel(0, 0, 0, false);

    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(L, 0, 0));
    EXPECT_TRUE(svo.get_voxel(2 * L - 1, L - 1, L - 1));
    EXPECT_EQ(svo.node_count(), 3u);
}
