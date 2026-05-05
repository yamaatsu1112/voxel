#include <algo/svt/svo_hamming.hpp>
#include <gtest/gtest.h>

TEST(HammingSvo, DefaultsToEmpty) {
    algo::svt::HammingSVO svo;
    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(10, 6, 19));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(HammingSvo, UsesConstructorMaxDepthForBounds) {
    algo::svt::HammingSVO svo(2);

    EXPECT_EQ(svo.max_depth(), 2u);
    EXPECT_EQ(svo.world_voxel_count(), algo::svt::LEAF_VOXEL_COUNT << 2);
    EXPECT_NO_THROW(svo.set_voxel(15, 15, 15, true));
    EXPECT_THROW(svo.set_voxel(16, 0, 0, true), std::out_of_range);
}

TEST(HammingSvo, SetAndResetVoxel) {
    algo::svt::HammingSVO svo;

    svo.set_voxel(1, 2, 3, true);
    EXPECT_TRUE(svo.get_voxel(1, 2, 3));
    EXPECT_EQ(svo.leaf_count(), 1u);

    svo.set_voxel(1, 2, 3, false);
    EXPECT_FALSE(svo.get_voxel(1, 2, 3));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(HammingSvo, SharesLeafWhenCanonicalDataMatches) {
    algo::svt::HammingSVO svo;

    // First leaf: single bit inside leaf 0
    const uint32_t first_x = 0;
    const uint32_t first_y = 0;
    const uint32_t first_z = 0;
    svo.set_voxel(first_x, first_y, first_z, true);
    EXPECT_TRUE(svo.get_voxel(first_x, first_y, first_z));
    EXPECT_EQ(svo.leaf_count(), 1u);

    // Second leaf: single bit inside the adjacent leaf in x direction
    const uint32_t second_x = algo::svt::LEAF_VOXEL_COUNT + 1;
    const uint32_t second_y = 0;
    const uint32_t second_z = 0;
    svo.set_voxel(second_x, second_y, second_z, true);
    EXPECT_TRUE(svo.get_voxel(second_x, second_y, second_z));

    // Both leaves are single-bit errors of the same canonical pattern, so
    // only one shared canonical leaf entry should exist.
    EXPECT_EQ(svo.leaf_count(), 1u);

    // Clearing the first should keep the shared leaf alive for the second.
    svo.set_voxel(first_x, first_y, first_z, false);
    EXPECT_FALSE(svo.get_voxel(first_x, first_y, first_z));
    EXPECT_TRUE(svo.get_voxel(second_x, second_y, second_z));
    EXPECT_EQ(svo.leaf_count(), 1u);

    // Once the second leaf is cleared, the shared canonical entry should be
    // released.
    svo.set_voxel(second_x, second_y, second_z, false);
    EXPECT_FALSE(svo.get_voxel(second_x, second_y, second_z));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(HammingSvo, CollapsesEmptyLeafParentWhenAllLeavesCleared) {
    algo::svt::HammingSVO svo;

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    // Place one voxel in each of 8 leaves under one LeafParent.
    // One LeafParent covers a (2L)^3 = 8^3 region, with 8 leaf children.
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * L, y * L, z * L, true);
            }
        }
    }

    const std::size_t node_count_with_leaves = svo.node_count();
    EXPECT_GT(svo.leaf_count(), 0u);

    // Clear all voxels so every leaf becomes empty.
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
    // The LeafParent should be freed, reducing node count.
    EXPECT_LT(svo.node_count(), node_count_with_leaves);
}

TEST(HammingSvo, CollapsesIntermediateNodeWhenAllLeafParentsBecomEmpty) {
    algo::svt::HammingSVO svo;

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    // Place voxels in leaves under different LeafParents that share a common
    // Node parent. Each LeafParent covers (2L)^3 = 8^3. A Node at depth
    // MAX_DEPTH-2 covers (4L)^3 = 16^3 with 8 LeafParent children.
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * 2 * L, y * 2 * L, z * 2 * L, true);
            }
        }
    }

    const std::size_t node_count_before = svo.node_count();

    // Clear all voxels.
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * 2 * L, y * 2 * L, z * 2 * L, false);
            }
        }
    }

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                EXPECT_FALSE(svo.get_voxel(x * 2 * L, y * 2 * L, z * 2 * L));
            }
        }
    }

    EXPECT_EQ(svo.leaf_count(), 0u);
    // The intermediate Node and all LeafParents should be freed.
    EXPECT_LT(svo.node_count(), node_count_before);
    // Only the root node should remain.
    EXPECT_EQ(svo.node_count(), 1u);
}
