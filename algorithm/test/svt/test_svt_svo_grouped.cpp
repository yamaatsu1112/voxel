#include <algo/svt/svo_grouped.hpp>
#include <gtest/gtest.h>

TEST(SvoGrouped, DefaultsToEmpty) {
    algo::svt::SVOGrouped svo;
    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(17, 9, 31));
}

TEST(SvoGrouped, UsesConstructorMaxDepthForBounds) {
    algo::svt::SVOGrouped svo(2);

    EXPECT_EQ(svo.max_depth(), 2u);
    EXPECT_EQ(svo.world_voxel_count(), algo::svt::LEAF_VOXEL_COUNT << 2);
    EXPECT_NO_THROW(svo.set_voxel(15, 15, 15, true));
    EXPECT_THROW(svo.set_voxel(16, 0, 0, true), std::out_of_range);
}

TEST(SvoGrouped, AllocatesLeavesInEightLeafBlocks) {
    algo::svt::SVOGrouped svo;

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    svo.set_voxel(0, 0, 0, true);
    EXPECT_EQ(svo.leaf_count(), 8u);

    svo.set_voxel(L, 0, 0, true);
    EXPECT_EQ(svo.leaf_count(), 8u);
    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(L, 0, 0));
}

TEST(SvoGrouped, SetAndResetVoxelReclaimsLeafBlock) {
    algo::svt::SVOGrouped svo;

    svo.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 8u);

    svo.set_voxel(5, 6, 7, false);
    EXPECT_FALSE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(SvoGrouped, CollapsesFullyFilledLeafBlockBackIntoParentBit) {
    algo::svt::SVOGrouped svo;

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    for (uint32_t leaf_z = 0; leaf_z < 2; ++leaf_z) {
        for (uint32_t leaf_y = 0; leaf_y < 2; ++leaf_y) {
            for (uint32_t leaf_x = 0; leaf_x < 2; ++leaf_x) {
                for (uint32_t z = 0; z < L; ++z) {
                    for (uint32_t y = 0; y < L; ++y) {
                        for (uint32_t x = 0; x < L; ++x) {
                            svo.set_voxel(leaf_x * L + x,
                                          leaf_y * L + y,
                                          leaf_z * L + z,
                                          true);
                        }
                    }
                }
            }
        }
    }

    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(2 * L - 1, 2 * L - 1, 2 * L - 1));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(SvoGrouped, CollapsesIntermediateNodeWhenEightLeavesBecomeEmpty) {
    algo::svt::SVOGrouped svo;

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * L, y * L, z * L, true);
            }
        }
    }

    EXPECT_EQ(svo.leaf_count(), 8u);
    const std::size_t node_count_with_leaves = svo.node_count();

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                svo.set_voxel(x * L, y * L, z * L, false);
            }
        }
    }

    EXPECT_EQ(svo.leaf_count(), 0u);
    EXPECT_LT(svo.node_count(), node_count_with_leaves);
    EXPECT_EQ(svo.node_count(), 1u);
}
