#include <algo/svt/svo_forwarding_packed_leaf.hpp>
#include <gtest/gtest.h>

TEST(SvoForwardingPackedLeaf, UsesLeafSizedWorldAtDepthZero) {
    algo::svt::SVOForwardingPackedLeaf svo(0);

    EXPECT_EQ(svo.max_depth(), 0u);
    EXPECT_EQ(svo.world_voxel_count(), 8u);
    EXPECT_NO_THROW(svo.set_voxel(7, 7, 7, true));
    EXPECT_THROW(svo.set_voxel(8, 0, 0, true), std::out_of_range);
}

TEST(SvoForwardingPackedLeaf, SetAndResetVoxelCollapsesBackToRoot) {
    algo::svt::SVOForwardingPackedLeaf svo(0);

    EXPECT_EQ(svo.node_count(), 9u);

    svo.set_voxel(1, 2, 3, true);
    EXPECT_TRUE(svo.get_voxel(1, 2, 3));
    EXPECT_EQ(svo.node_count(), 27u);

    svo.set_voxel(1, 2, 3, false);
    EXPECT_FALSE(svo.get_voxel(1, 2, 3));
    EXPECT_EQ(svo.node_count(), 9u);
}

TEST(SvoForwardingPackedLeaf, FullyFilledLeafBlockCollapsesIntoParentBits) {
    algo::svt::SVOForwardingPackedLeaf svo(0);

    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                svo.set_voxel(x, y, z, true);
            }
        }
    }

    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_TRUE(svo.get_voxel(3, 3, 3));
    EXPECT_EQ(svo.node_count(), 9u);
}

TEST(SvoForwardingPackedLeaf, SupportsForwardingAcrossMultiplePages) {
    algo::svt::SVOForwardingPackedLeafImpl<7> svo(2);

    const uint32_t extent = svo.world_voxel_count();
    for (uint32_t x = 0; x < extent; ++x) {
        for (uint32_t y = 0; y < extent; ++y) {
            for (uint32_t z = 0; z < extent; ++z) {
                if ((x + y + z) % 2 == 0) {
                    svo.set_voxel(x, y, z, true);
                }
            }
        }
    }

    EXPECT_GT(svo.page_count(), 1u);
    EXPECT_GT(svo.forwarded_cell_count(), 0u);
    EXPECT_TRUE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(0, 0, 1));
}
