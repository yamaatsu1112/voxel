#include <algo/svt/svo64_grouped.hpp>
#include <gtest/gtest.h>

TEST(Svo64Grouped, DefaultsToEmpty) {
    algo::svt::SVO64Grouped svo;
    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(17, 9, 31));
}

TEST(Svo64Grouped, UsesConstructorMaxDepthForBounds) {
    algo::svt::SVO64Grouped svo(2);

    EXPECT_EQ(svo.max_depth(), 2u);
    EXPECT_EQ(svo.world_voxel_count(), algo::svt::LEAF_VOXEL_COUNT << 4);
    EXPECT_NO_THROW(svo.set_voxel(63, 63, 63, true));
    EXPECT_THROW(svo.set_voxel(64, 0, 0, true), std::out_of_range);
}

TEST(Svo64Grouped, AllocatesLeavesInSixtyFourLeafBlocks) {
    algo::svt::SVO64Grouped svo(1);

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    svo.set_voxel(0, 0, 0, true);
    EXPECT_EQ(svo.leaf_count(), 64u);

    svo.set_voxel(3 * L, 0, 0, true);
    EXPECT_EQ(svo.leaf_count(), 64u);
    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(3 * L, 0, 0));
}

TEST(Svo64Grouped, SetAndResetVoxelReclaimsLeafBlock) {
    algo::svt::SVO64Grouped svo(1);

    svo.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 64u);

    svo.set_voxel(5, 6, 7, false);
    EXPECT_FALSE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(Svo64Grouped, CollapsesFullyFilledLeafBlockBackIntoParentBit) {
    algo::svt::SVO64Grouped svo(1);

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    for (uint32_t leaf_z = 0; leaf_z < 4; ++leaf_z) {
        for (uint32_t leaf_y = 0; leaf_y < 4; ++leaf_y) {
            for (uint32_t leaf_x = 0; leaf_x < 4; ++leaf_x) {
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
    EXPECT_TRUE(svo.get_voxel(4 * L - 1, 4 * L - 1, 4 * L - 1));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(Svo64Grouped, CollapsesIntermediateNodeWhenSixtyFourLeavesBecomeEmpty) {
    algo::svt::SVO64Grouped svo(2);

    constexpr uint32_t L = algo::svt::LEAF_VOXEL_COUNT;

    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                svo.set_voxel(x * L, y * L, z * L, true);
            }
        }
    }

    EXPECT_EQ(svo.leaf_count(), 64u);
    const std::size_t node_count_with_leaves = svo.node_count();

    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                svo.set_voxel(x * L, y * L, z * L, false);
            }
        }
    }

    EXPECT_EQ(svo.leaf_count(), 0u);
    EXPECT_LT(svo.node_count(), node_count_with_leaves);
    EXPECT_EQ(svo.node_count(), 1u);
}
