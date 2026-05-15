#pragma once

std::uint32_t terminal_request_representative_leaf_key_host(
    const algo::svt::cuda::TerminalRequest& request) {
    namespace cuda = algo::svt::cuda;
    if (cuda::terminal_request_is_brick(request))
        return cuda::terminal_request_brick(request).leafPrefix;

    const cuda::CellWriteRequest cell = cuda::terminal_request_cell(request);
    const std::uint32_t shift =
        (cuda::kMaxDepth - cell.level) * cuda::kGroupSizeExp;
    return static_cast<std::uint32_t>(cell.prefix << shift);
}

void copy_terminal_request_keys_to_workspace(
    const std::vector<algo::svt::cuda::TerminalRequest>& requests,
    algo::svt::cuda::detail::TerminalEditWorkspace& workspace) {
    std::vector<std::uint32_t> keys;
    keys.reserve(requests.size());
    for (const auto& request : requests)
        keys.push_back(terminal_request_representative_leaf_key_host(request));
    copy_to_device(workspace.request_keys, keys);
}

TEST(CudaTerminalEditRequests, PruneDropsCoveredBricksBeforeMergingMasks) {
    expect_cuda_device_or_skip();

    namespace cuda = algo::svt::cuda;
    std::vector<cuda::TerminalRequest> requests = {
        cuda::make_terminal_brick_request(
            {static_cast<std::uint32_t>(leaf_key(0u, 0u, 0u)), 1u}),
        cuda::make_terminal_cell_request({10u, 0u}),
        cuda::make_terminal_cell_request({9u, 0u}),
        cuda::make_terminal_brick_request(
            {static_cast<std::uint32_t>(leaf_key(2u, 0u, 0u)), 1u}),
        cuda::make_terminal_brick_request(
            {static_cast<std::uint32_t>(leaf_key(2u, 0u, 0u)), 2u}),
    };

    auto* d_requests = device_alloc<cuda::TerminalRequest>(requests.size());
    ASSERT_NE(d_requests, nullptr);
    copy_to_device(d_requests, requests);

    const auto workspace_size =
        cuda::detail::apply_terminal_edits_workspace_size<
            cuda::DefaultTerminalEditConfig::allocation,
            cuda::DefaultTerminalEditConfig::release>(
            0u, 0u, static_cast<std::uint32_t>(requests.size()));
    auto* workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);
    auto typed_workspace = cuda::detail::create_terminal_edit_workspace<
        cuda::DefaultTerminalEditConfig::allocation,
        cuda::DefaultTerminalEditConfig::release>(
        workspace, 0u, 0u, workspace_size);
    copy_terminal_request_keys_to_workspace(requests, typed_workspace);

    auto request_count = static_cast<std::uint32_t>(requests.size());
    auto count_sort_workspace =
        cuda::detail::create_terminal_count_sort_workspace(
            typed_workspace.phase_scratch, 0u,
            typed_workspace.request_capacity);
    ASSERT_EQ(algo::cuda::sort::sort_by_key(
                  typed_workspace.request_keys,
                  algo::cuda::sort::value_arrays(d_requests), request_count,
                  count_sort_workspace.sort_workspace,
                  count_sort_workspace.sort_workspace_size, nullptr),
              cudaSuccess);
    ASSERT_EQ(cuda::detail::prune_terminal_requests(
                  d_requests, typed_workspace.pruned_requests, request_count,
                  typed_workspace.request_keys,
                  cuda::detail::create_terminal_prune_workspace(
                      typed_workspace.phase_scratch,
                      typed_workspace.request_capacity),
                  nullptr),
              cudaSuccess);
    sync_cuda();

    const auto pruned =
        copy_from_device(typed_workspace.pruned_requests, request_count);
    ASSERT_EQ(pruned.size(), 2u);
    EXPECT_TRUE(cuda::terminal_request_is_cell(pruned[0]));
    const cuda::CellWriteRequest cell = cuda::terminal_request_cell(pruned[0]);
    EXPECT_EQ(cell.level, 9u);
    EXPECT_EQ(cell.prefix, 0u);
    EXPECT_TRUE(cuda::terminal_request_is_brick(pruned[1]));
    const cuda::TerminalBrickMask brick =
        cuda::terminal_request_brick(pruned[1]);
    EXPECT_EQ(brick.leafPrefix, leaf_key(2u, 0u, 0u));
    EXPECT_EQ(brick.mask64, 3u);

    cudaFree(workspace);
    cudaFree(d_requests);
}

TEST(CudaTerminalEditRequests, PruneKeepsCanonicalAncestorOnly) {
    expect_cuda_device_or_skip();

    namespace cuda = algo::svt::cuda;
    std::vector<cuda::TerminalRequest> requests = {
        cuda::make_terminal_cell_request({10u, 0u}),
        cuda::make_terminal_cell_request({8u, 0u}),
        cuda::make_terminal_cell_request({9u, 1u}),
        cuda::make_terminal_cell_request({7u, 0u}),
    };

    auto* d_requests = device_alloc<cuda::TerminalRequest>(requests.size());
    ASSERT_NE(d_requests, nullptr);
    copy_to_device(d_requests, requests);

    const auto workspace_size =
        cuda::detail::apply_terminal_edits_workspace_size<
            cuda::DefaultTerminalEditConfig::allocation,
            cuda::DefaultTerminalEditConfig::release>(
            0u, 0u, static_cast<std::uint32_t>(requests.size()));
    auto* workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);
    auto typed_workspace = cuda::detail::create_terminal_edit_workspace<
        cuda::DefaultTerminalEditConfig::allocation,
        cuda::DefaultTerminalEditConfig::release>(
        workspace, 0u, 0u, workspace_size);
    copy_terminal_request_keys_to_workspace(requests, typed_workspace);

    auto request_count = static_cast<std::uint32_t>(requests.size());
    auto count_sort_workspace =
        cuda::detail::create_terminal_count_sort_workspace(
            typed_workspace.phase_scratch, 0u,
            typed_workspace.request_capacity);
    ASSERT_EQ(algo::cuda::sort::sort_by_key(
                  typed_workspace.request_keys,
                  algo::cuda::sort::value_arrays(d_requests), request_count,
                  count_sort_workspace.sort_workspace,
                  count_sort_workspace.sort_workspace_size, nullptr),
              cudaSuccess);
    ASSERT_EQ(cuda::detail::prune_terminal_requests(
                  d_requests, typed_workspace.pruned_requests, request_count,
                  typed_workspace.request_keys,
                  cuda::detail::create_terminal_prune_workspace(
                      typed_workspace.phase_scratch,
                      typed_workspace.request_capacity),
                  nullptr),
              cudaSuccess);
    sync_cuda();

    const auto pruned =
        copy_from_device(typed_workspace.pruned_requests, request_count);
    ASSERT_EQ(pruned.size(), 1u);
    EXPECT_TRUE(cuda::terminal_request_is_cell(pruned[0]));
    const cuda::CellWriteRequest cell = cuda::terminal_request_cell(pruned[0]);
    EXPECT_EQ(cell.level, 7u);
    EXPECT_EQ(cell.prefix, 0u);

    cudaFree(workspace);
    cudaFree(d_requests);
}
