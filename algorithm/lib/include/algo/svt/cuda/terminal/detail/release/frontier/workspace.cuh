#pragma once

#include <algo/svt/cuda/terminal/detail/release/workspace.cuh>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct TerminalFrontierReleaseWorkspace {
  TerminalReleaseWorkspaceBase base;
  std::uint32_t *node_free_counts;
  std::uint32_t *leaf_free_counts;
};

inline std::size_t
terminal_frontier_release_workspace_size(std::uint32_t request_capacity) {
  std::size_t offset = terminal_release_workspace_base_size(request_capacity);
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  return offset;
}

inline TerminalFrontierReleaseWorkspace
create_terminal_frontier_release_workspace(
    void *workspace, TerminalDetachedChild *detached_children,
    std::uint32_t *detached_child_count, std::uint32_t request_capacity) {
  TerminalFrontierReleaseWorkspace result{};
  result.base = create_terminal_release_workspace_base(
      workspace, detached_children, detached_child_count, request_capacity);
  std::size_t offset = terminal_release_workspace_base_size(request_capacity);
  reserve_array(workspace, request_capacity, offset, result.node_free_counts);
  reserve_array(workspace, request_capacity, offset, result.leaf_free_counts);
  return result;
}

} // namespace algo::svt::cuda::detail
