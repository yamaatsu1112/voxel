#include <algo/svt/svo.hpp>
#include <gtest/gtest.h>

TEST(Svo, DefaultsToEmpty) {
    algo::svt::SVO svo;
    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(17, 9, 31));
}

TEST(Svo, UsesConstructorMaxDepthForBounds) {
    algo::svt::SVO svo(2);

    EXPECT_EQ(svo.max_depth(), 2u);
    EXPECT_EQ(svo.world_voxel_count(), algo::svt::LEAF_VOXEL_COUNT << 2);
    EXPECT_NO_THROW(svo.set_voxel(15, 15, 15, true));
    EXPECT_THROW(svo.set_voxel(16, 0, 0, true), std::out_of_range);
}

TEST(Svo, SetAndResetVoxel) {
    algo::svt::SVO svo;

    svo.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 1u);

    svo.set_voxel(5, 6, 7, false);
    EXPECT_FALSE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(Svo, CollapsesFullyFilledLeafBackIntoParentBit) {
    algo::svt::SVO svo;

    for (uint32_t z = 0; z < algo::svt::LEAF_VOXEL_COUNT; ++z) {
        for (uint32_t y = 0; y < algo::svt::LEAF_VOXEL_COUNT; ++y) {
            for (uint32_t x = 0; x < algo::svt::LEAF_VOXEL_COUNT; ++x) {
                svo.set_voxel(x, y, z, true);
            }
        }
    }

    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(3, 3, 3));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(Svo, CollapsesIntermediateNodeWhenEightLeavesBecomeFull) {
    algo::svt::SVO svo;

    svo.set_voxel(0, 0, 0, true);
    const std::size_t node_count_before = svo.node_count();

    for (uint32_t z = 0; z < 2 * algo::svt::LEAF_VOXEL_COUNT; ++z) {
        for (uint32_t y = 0; y < 2 * algo::svt::LEAF_VOXEL_COUNT; ++y) {
            for (uint32_t x = 0; x < 2 * algo::svt::LEAF_VOXEL_COUNT; ++x) {
                svo.set_voxel(x, y, z, true);
            }
        }
    }

    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(7, 7, 7));
    EXPECT_EQ(svo.leaf_count(), 0u);
    EXPECT_LT(svo.node_count(), node_count_before + 7u);
    EXPECT_EQ(svo.node_count(), static_cast<std::size_t>(svo.max_depth() - 1));
}

TEST(Svo, CollapsesFullyEmptiedLeafBackIntoParentBit) {
    algo::svt::SVO svo;

    // Fill several voxels inside one leaf.
    svo.set_voxel(0, 0, 0, true);
    svo.set_voxel(1, 0, 0, true);
    svo.set_voxel(0, 1, 0, true);
    EXPECT_EQ(svo.leaf_count(), 1u);

    const std::size_t node_count_before = svo.node_count();

    // Clear all voxels so the leaf becomes completely empty.
    svo.set_voxel(0, 0, 0, false);
    svo.set_voxel(1, 0, 0, false);
    svo.set_voxel(0, 1, 0, false);

    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(1, 0, 0));
    EXPECT_FALSE(svo.get_voxel(0, 1, 0));
    EXPECT_EQ(svo.leaf_count(), 0u);
    EXPECT_LT(svo.node_count(), node_count_before);
}

TEST(Svo, CollapsesIntermediateNodeWhenEightLeavesBecomEmpty) {
    algo::svt::SVO svo;

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    // Place one voxel in each of the 8 leaves under one intermediate node.
    // The 8 leaves cover the 8 sub-cubes of a (2L)^3 region.
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * L, y * L, z * L, true);
            }
        }
    }

    EXPECT_EQ(svo.leaf_count(), 8u);
    const std::size_t node_count_with_leaves = svo.node_count();

    // Clear all 8 voxels so every leaf becomes empty.
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * L, y * L, z * L, false);
            }
        }
    }

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                EXPECT_FALSE(svo.get_voxel(x * L, y * L, z * L));
            }
        }
    }

    EXPECT_EQ(svo.leaf_count(), 0u);
    // The intermediate node that held the 8 leaves should also be freed.
    EXPECT_LT(svo.node_count(), node_count_with_leaves);
    // Everything collapses back to just the root node.
    EXPECT_EQ(svo.node_count(), 1u);
}

TEST(Svo, PreservesMixedParentWhenSiblingLeafBecomesEmpty) {
    algo::svt::SVO svo(2);

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    for (uint32_t z = 0; z < L; ++z) {
        for (uint32_t y = 0; y < L; ++y) {
            for (uint32_t x = 0; x < L; ++x) {
                svo.set_voxel(L + x, y, z, true);
            }
        }
    }

    svo.set_voxel(0, 0, 0, true);
    EXPECT_EQ(svo.node_count(), 2u);
    EXPECT_EQ(svo.leaf_count(), 1u);

    svo.set_voxel(0, 0, 0, false);

    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(L, 0, 0));
    EXPECT_TRUE(svo.get_voxel(2 * L - 1, L - 1, L - 1));
    EXPECT_EQ(svo.node_count(), 2u);
    EXPECT_EQ(svo.leaf_count(), 0u);
}
