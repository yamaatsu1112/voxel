#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct TerminalReleaseWorkspaceBase {
  TerminalDetachedChild *detached_children;
  std::uint32_t *detached_child_count;
  TerminalDetachedChild *frontier;
  TerminalDetachedChild *next_frontier;
  std::uint32_t *emit_counts;
  std::uint32_t *frontier_count;
  std::uint32_t *next_frontier_count;
  cudaError_t *status;
  AllocationState *allocation_state;
  std::uint32_t capacity;
  void *scan_workspace;
  std::size_t scan_workspace_size;
};

inline std::size_t
terminal_release_workspace_base_size(std::uint32_t request_capacity) {
  std::size_t offset = 0;
  offset = detail::align_up<TerminalDetachedChild>(offset);
  offset += sizeof(TerminalDetachedChild) * request_capacity;
  offset = detail::align_up<TerminalDetachedChild>(offset);
  offset += sizeof(TerminalDetachedChild) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);
  offset = detail::align_up<cudaError_t>(offset);
  offset += sizeof(cudaError_t);
  offset = detail::align_up<AllocationState>(offset);
  offset += sizeof(AllocationState);
  offset = detail::align_up<std::max_align_t>(offset);
  offset += algo::cuda::scan::required_workspace_size<std::uint32_t>(
      request_capacity);
  return offset;
}

inline TerminalReleaseWorkspaceBase create_terminal_release_workspace_base(
    void *workspace, TerminalDetachedChild *detached_children,
    std::uint32_t *detached_child_count, std::uint32_t request_capacity) {
  TerminalReleaseWorkspaceBase result{};
  result.detached_children = detached_children;
  result.detached_child_count = detached_child_count;
  result.capacity = request_capacity;
  std::size_t offset = 0;
  reserve_array(workspace, request_capacity, offset, result.frontier);
  reserve_array(workspace, request_capacity, offset, result.next_frontier);
  reserve_array(workspace, request_capacity, offset, result.emit_counts);
  reserve_array(workspace, 1u, offset, result.frontier_count);
  reserve_array(workspace, 1u, offset, result.next_frontier_count);
  reserve_array(workspace, 1u, offset, result.status);
  reserve_array(workspace, 1u, offset, result.allocation_state);
  result.scan_workspace_size =
      algo::cuda::scan::required_workspace_size<std::uint32_t>(
          request_capacity);
  reserve_workspace<std::max_align_t>(workspace, result.scan_workspace_size,
                                      offset, result.scan_workspace);
  return result;
}

} // namespace algo::svt::cuda::detail
