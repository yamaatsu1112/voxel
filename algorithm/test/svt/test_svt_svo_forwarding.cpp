#include <algo/svt/svo_forwarding.hpp>
#include <gtest/gtest.h>

TEST(SvoForwarding, DefaultsToEmpty) {
    algo::svt::SVOForwarding svo;
    EXPECT_FALSE(svo.get_voxel(0, 0, 0));
    EXPECT_FALSE(svo.get_voxel(17, 9, 31));
}

TEST(SvoForwarding, UsesConstructorMaxDepthForBounds) {
    algo::svt::SVOForwarding svo(2);

    EXPECT_EQ(svo.max_depth(), 2u);
    EXPECT_EQ(svo.world_voxel_count(), 8u);
    EXPECT_NO_THROW(svo.set_voxel(7, 7, 7, true));
    EXPECT_THROW(svo.set_voxel(8, 0, 0, true), std::out_of_range);
}

TEST(SvoForwarding, SetAndResetVoxel) {
    algo::svt::SVOForwarding svo;

    svo.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 0u);

    svo.set_voxel(5, 6, 7, false);
    EXPECT_FALSE(svo.get_voxel(5, 6, 7));
    EXPECT_EQ(svo.leaf_count(), 0u);
}

TEST(SvoForwarding, CollapsesFullyFilledTerminalNodeBackIntoParentBit) {
    algo::svt::SVOForwarding svo;
    const std::size_t node_count_before = svo.node_count();

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
    EXPECT_GT(svo.node_count(), node_count_before);
    EXPECT_LT(svo.node_count(), node_count_before + svo.max_depth() * 8u);
}

TEST(SvoForwarding, CollapsesIntermediateNodeWhenEightTerminalNodesBecomeEmpty) {
    algo::svt::SVOForwarding svo;

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

TEST(SvoForwarding, ObservesForwardingAndAdditionalPagesAfterOverflow) {
    algo::svt::SVOForwardingImpl<7> svo(5);

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

TEST(SvoForwarding, ClearingForwardedTreeKeepsCountsSane) {
    algo::svt::SVOForwardingImpl<7> svo(3);

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

    for (uint32_t x = 0; x < extent; ++x) {
        for (uint32_t y = 0; y < extent; ++y) {
            for (uint32_t z = 0; z < extent; ++z) {
                if ((x + y + z) % 2 == 0) {
                    svo.set_voxel(x, y, z, false);
                }
            }
        }
    }

    EXPECT_EQ(svo.forwarded_cell_count(), 0u);
    EXPECT_EQ(svo.node_count(), 1u);
}
