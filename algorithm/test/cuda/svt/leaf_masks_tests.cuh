#pragma once

TEST(CudaSvt, ConvertsVoxelEditsToLeafMasks) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::LeafMask;
    using algo::svt::cuda::VoxelEdit;

    const std::vector<VoxelEdit> edits = {
        {0, 0, 0}, {1, 0, 0}, {0, 0, 2}, {4, 0, 0}, {1023, 1023, 1023}};
    auto *d_edits = device_alloc<VoxelEdit>(edits.size());
    auto *d_leaf_masks = device_alloc<LeafMask>(edits.size());
    auto *d_leaf_count = device_alloc<std::uint32_t>(1);
    ASSERT_NE(d_edits, nullptr);
    ASSERT_NE(d_leaf_masks, nullptr);
    ASSERT_NE(d_leaf_count, nullptr);

    copy_to_device(d_edits, edits);
    const auto workspace_size =
        algo::svt::cuda::voxel_edits_to_leaf_masks_workspace_size(
            static_cast<std::uint32_t>(edits.size()));
    auto *workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    ASSERT_EQ(algo::svt::cuda::voxel_edits_to_leaf_masks(
                  d_edits, static_cast<std::uint32_t>(edits.size()),
                  d_leaf_masks, d_leaf_count, workspace, workspace_size),
              cudaSuccess);
    sync_cuda();

    const auto leaf_count = copy_scalar_from_device(d_leaf_count);
    ASSERT_EQ(leaf_count, 3u);
    std::vector<LeafMask> masks = copy_from_device(d_leaf_masks, leaf_count);
    std::sort(masks.begin(), masks.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.leaf_key < rhs.leaf_key;
    });

    EXPECT_EQ(masks[0].leaf_key, expected_leaf_key(0, 0, 0));
    EXPECT_EQ(masks[0].voxel_data_low, 0b11u);
    EXPECT_EQ(masks[0].voxel_data_high, 1u);

    EXPECT_EQ(masks[1].leaf_key, expected_leaf_key(4, 0, 0));
    EXPECT_EQ(masks[1].voxel_data_low, 1u);
    EXPECT_EQ(masks[1].voxel_data_high, 0u);

    EXPECT_EQ(masks[2].leaf_key, expected_leaf_key(1023, 1023, 1023));
    EXPECT_EQ(masks[2].voxel_data_low, 0u);
    EXPECT_EQ(masks[2].voxel_data_high, 1u << 31u);

    cudaFree(workspace);
    cudaFree(d_leaf_count);
    cudaFree(d_leaf_masks);
    cudaFree(d_edits);
}

TEST(CudaSvt, RejectsOutOfBoundsVoxelEdits) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::LeafMask;
    using algo::svt::cuda::VoxelEdit;

    const std::vector<VoxelEdit> edits = {
        {0, 0, 0}, {algo::svt::cuda::kWorldVoxelCount, 0, 0}};
    auto *d_edits = device_alloc<VoxelEdit>(edits.size());
    auto *d_leaf_masks = device_alloc<LeafMask>(edits.size());
    auto *d_leaf_count = device_alloc<std::uint32_t>(1);
    ASSERT_NE(d_edits, nullptr);
    ASSERT_NE(d_leaf_masks, nullptr);
    ASSERT_NE(d_leaf_count, nullptr);

    copy_to_device(d_edits, edits);
    const auto workspace_size =
        algo::svt::cuda::voxel_edits_to_leaf_masks_workspace_size(
            static_cast<std::uint32_t>(edits.size()));
    auto *workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    ASSERT_EQ(algo::svt::cuda::voxel_edits_to_leaf_masks(
                  d_edits, static_cast<std::uint32_t>(edits.size()),
                  d_leaf_masks, d_leaf_count, workspace, workspace_size),
              cudaSuccess);
    sync_cuda();

    const auto leaf_count = copy_scalar_from_device(d_leaf_count);
    ASSERT_EQ(leaf_count, 1u);
    const auto masks = copy_from_device(d_leaf_masks, leaf_count);
    EXPECT_EQ(masks[0].leaf_key, expected_leaf_key(0, 0, 0));
    EXPECT_EQ(masks[0].voxel_data_low, 1u);
    EXPECT_EQ(masks[0].voxel_data_high, 0u);

    cudaFree(workspace);
    cudaFree(d_leaf_count);
    cudaFree(d_leaf_masks);
    cudaFree(d_edits);
}

TEST(CudaSvt, EmitsLeafMasksInSvoTraversalOrder) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::LeafMask;
    using algo::svt::cuda::VoxelEdit;

    const std::vector<VoxelEdit> edits = {
        {8, 0, 0}, {0, 0, 4}, {0, 4, 0}, {0, 0, 0}};
    auto *d_edits = device_alloc<VoxelEdit>(edits.size());
    auto *d_leaf_masks = device_alloc<LeafMask>(edits.size());
    auto *d_leaf_count = device_alloc<std::uint32_t>(1);
    ASSERT_NE(d_edits, nullptr);
    ASSERT_NE(d_leaf_masks, nullptr);
    ASSERT_NE(d_leaf_count, nullptr);

    copy_to_device(d_edits, edits);
    const auto workspace_size =
        algo::svt::cuda::voxel_edits_to_leaf_masks_workspace_size(
            static_cast<std::uint32_t>(edits.size()));
    auto *workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    ASSERT_EQ(algo::svt::cuda::voxel_edits_to_leaf_masks(
                  d_edits, static_cast<std::uint32_t>(edits.size()),
                  d_leaf_masks, d_leaf_count, workspace, workspace_size),
              cudaSuccess);
    sync_cuda();

    const auto leaf_count = copy_scalar_from_device(d_leaf_count);
    ASSERT_EQ(leaf_count, 4u);
    const auto masks = copy_from_device(d_leaf_masks, leaf_count);

    EXPECT_LT(expected_leaf_key(0, 0, 0), expected_leaf_key(0, 4, 0));
    EXPECT_LT(expected_leaf_key(0, 4, 0), expected_leaf_key(0, 0, 4));
    EXPECT_LT(expected_leaf_key(0, 0, 4), expected_leaf_key(8, 0, 0));
    EXPECT_EQ(masks[0].leaf_key, expected_leaf_key(0, 0, 0));
    EXPECT_EQ(masks[1].leaf_key, expected_leaf_key(0, 4, 0));
    EXPECT_EQ(masks[2].leaf_key, expected_leaf_key(0, 0, 4));
    EXPECT_EQ(masks[3].leaf_key, expected_leaf_key(8, 0, 0));

    cudaFree(workspace);
    cudaFree(d_leaf_count);
    cudaFree(d_leaf_masks);
    cudaFree(d_edits);
}

TEST(CudaSvt, BuildLeafMasksEmitsExpectedMasks) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    using algo::svt::cuda::LeafMask;
    using algo::svt::cuda::VoxelEdit;

    const std::vector<VoxelEdit> edits = {
        {0, 0, 0}, {1, 0, 0}, {0, 0, 2}, {4, 0, 0}, {1023, 1023, 1023}};
    auto *d_edits = device_alloc<VoxelEdit>(edits.size());
    auto *d_leaf_masks = device_alloc<LeafMask>(edits.size());
    auto *d_leaf_count = device_alloc<std::uint32_t>(1);
    ASSERT_NE(d_edits, nullptr);
    ASSERT_NE(d_leaf_masks, nullptr);
    ASSERT_NE(d_leaf_count, nullptr);

    copy_to_device(d_edits, edits);
    const auto workspace_size =
        algo::svt::cuda::voxel_edits_to_leaf_masks_workspace_size(
            static_cast<std::uint32_t>(edits.size()));
    auto *workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    ASSERT_EQ((algo::svt::cuda::voxel_edits_to_leaf_masks(
                  d_edits, static_cast<std::uint32_t>(edits.size()),
                  d_leaf_masks, d_leaf_count, workspace, workspace_size)),
              cudaSuccess);
    sync_cuda();

    const auto leaf_count = copy_scalar_from_device(d_leaf_count);
    ASSERT_EQ(leaf_count, 3u);
    std::vector<LeafMask> masks = copy_from_device(d_leaf_masks, leaf_count);
    std::sort(masks.begin(), masks.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.leaf_key < rhs.leaf_key;
    });

    EXPECT_EQ(masks[0].leaf_key, expected_leaf_key(0, 0, 0));
    EXPECT_EQ(masks[0].voxel_data_low, 0b11u);
    EXPECT_EQ(masks[0].voxel_data_high, 1u);

    EXPECT_EQ(masks[1].leaf_key, expected_leaf_key(4, 0, 0));
    EXPECT_EQ(masks[1].voxel_data_low, 1u);
    EXPECT_EQ(masks[1].voxel_data_high, 0u);

    EXPECT_EQ(masks[2].leaf_key, expected_leaf_key(1023, 1023, 1023));
    EXPECT_EQ(masks[2].voxel_data_low, 0u);
    EXPECT_EQ(masks[2].voxel_data_high, 1u << 31u);

    cudaFree(workspace);
    cudaFree(d_leaf_count);
    cudaFree(d_leaf_masks);
    cudaFree(d_edits);
}
