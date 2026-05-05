#include <algo/svt/hash_dag.hpp>
#include <gtest/gtest.h>

TEST(HashDAG, DefaultsToEmpty) {
    algo::svt::HashDAG dag;

    EXPECT_FALSE(dag.get_voxel(0, 0, 0));
    EXPECT_FALSE(dag.get_voxel(17, 9, 31));
    EXPECT_EQ(dag.depth(), algo::svt::DEFAULT_MAX_DEPTH);
    EXPECT_EQ(dag.world_voxel_count(), algo::svt::LEAF_VOXEL_COUNT << dag.depth());
}

TEST(HashDAG, UsesConstructorDepthForBounds) {
    algo::svt::HashDAG dag(2);

    EXPECT_EQ(dag.depth(), 2u);
    EXPECT_EQ(dag.world_voxel_count(), 16u);
    EXPECT_NO_THROW(dag.set_voxel(15, 15, 15, true));
    EXPECT_THROW(dag.set_voxel(16, 0, 0, true), std::out_of_range);
}

TEST(HashDAG, SetAndResetVoxel) {
    algo::svt::HashDAG dag(2);
    const std::size_t initial_nodes = dag.node_count();
    const std::size_t initial_leaves = dag.leaf_count();

    dag.set_voxel(5, 6, 7, true);
    EXPECT_TRUE(dag.get_voxel(5, 6, 7));
    EXPECT_GT(dag.node_count(), initial_nodes);
    EXPECT_EQ(dag.leaf_count(), initial_leaves + 1u);

    dag.set_voxel(5, 6, 7, false);
    EXPECT_FALSE(dag.get_voxel(5, 6, 7));
    EXPECT_EQ(dag.root(), 0u);
}

TEST(HashDAG, ReusesIdenticalLeafData) {
    algo::svt::HashDAG dag(2);
    const std::size_t initial_leaves = dag.leaf_count();

    dag.set_voxel(0, 0, 0, true);
    EXPECT_EQ(dag.leaf_count(), initial_leaves + 1u);

    dag.set_voxel(4, 0, 0, true);
    EXPECT_TRUE(dag.get_voxel(0, 0, 0));
    EXPECT_TRUE(dag.get_voxel(4, 0, 0));
    EXPECT_EQ(dag.leaf_count(), initial_leaves + 1u);
}

TEST(HashDAG, EditingSharedLeafDoesNotChangeOtherInstance) {
    algo::svt::HashDAG dag(2);

    dag.set_voxel(0, 0, 0, true);
    dag.set_voxel(4, 0, 0, true);
    dag.set_voxel(0, 0, 0, false);

    EXPECT_FALSE(dag.get_voxel(0, 0, 0));
    EXPECT_TRUE(dag.get_voxel(4, 0, 0));
}

TEST(HashDAG, FillAndClearWholeVolumeUseReservedStates) {
    algo::svt::HashDAG dag(2);
    const std::size_t initial_nodes = dag.node_count();
    const std::size_t initial_leaves = dag.leaf_count();

    dag.fill_box({0, 0, 0, dag.world_voxel_count(), dag.world_voxel_count(),
                  dag.world_voxel_count()});

    EXPECT_TRUE(dag.get_voxel(0, 0, 0));
    EXPECT_TRUE(dag.get_voxel(15, 15, 15));
    EXPECT_EQ(dag.node_count(), initial_nodes);
    EXPECT_EQ(dag.leaf_count(), initial_leaves);

    dag.clear_box({0, 0, 0, dag.world_voxel_count(), dag.world_voxel_count(),
                   dag.world_voxel_count()});

    EXPECT_FALSE(dag.get_voxel(0, 0, 0));
    EXPECT_FALSE(dag.get_voxel(15, 15, 15));
    EXPECT_EQ(dag.root(), 0u);
}

TEST(HashDAG, KeepsOnlyCurrentRoot) {
    algo::svt::HashDAG dag(2);

    dag.set_voxel(1, 2, 3, true);
    EXPECT_TRUE(dag.get_voxel(1, 2, 3));
    const uint32_t filled_root = dag.root();

    dag.set_voxel(1, 2, 3, false);
    EXPECT_FALSE(dag.get_voxel(1, 2, 3));
    EXPECT_NE(dag.root(), filled_root);

    dag.set_voxel(1, 2, 3, true);
    EXPECT_TRUE(dag.get_voxel(1, 2, 3));
    EXPECT_EQ(dag.root(), filled_root);
}

TEST(HashDAG, CollectGarbageDropsUnreachableEntries) {
    algo::svt::HashDAG dag(2);
    const std::size_t initial_nodes = dag.node_count();
    const std::size_t initial_leaves = dag.leaf_count();

    dag.set_voxel(1, 2, 3, true);
    EXPECT_GT(dag.node_count(), initial_nodes);
    EXPECT_GT(dag.leaf_count(), initial_leaves);

    dag.set_voxel(1, 2, 3, false);
    EXPECT_EQ(dag.root(), 0u);
    EXPECT_GT(dag.node_count(), initial_nodes);

    dag.collect_garbage();

    EXPECT_EQ(dag.root(), 0u);
    EXPECT_EQ(dag.node_count(), initial_nodes);
    EXPECT_EQ(dag.leaf_count(), initial_leaves);
}

TEST(HashDAG, CollectGarbageKeepsCurrentRootAndReservedFullSubtrees) {
    algo::svt::HashDAG dag(2);

    dag.set_voxel(0, 0, 0, true);
    dag.set_voxel(15, 15, 15, true);

    dag.collect_garbage();

    EXPECT_TRUE(dag.get_voxel(0, 0, 0));
    EXPECT_TRUE(dag.get_voxel(15, 15, 15));
    EXPECT_FALSE(dag.get_voxel(7, 7, 7));

    dag.fill_box({0, 0, 0, dag.world_voxel_count(), dag.world_voxel_count(),
                  dag.world_voxel_count()});

    EXPECT_TRUE(dag.get_voxel(7, 7, 7));
    EXPECT_TRUE(dag.get_voxel(15, 15, 15));
}
