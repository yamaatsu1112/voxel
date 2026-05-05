#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct TerminalCollapseWorkspace {
  std::uint32_t *request_keys;
  std::uint32_t *unique_offsets;
  AllocationRequest *unique_requests;
  std::uint32_t *unique_count;
  AllocationState *allocation_state;
  void *scan_workspace;
  std::size_t scan_workspace_size;
};

inline std::size_t
terminal_collapse_workspace_size(std::uint32_t request_capacity) {
  std::size_t offset = 0;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<AllocationRequest>(offset);
  offset += sizeof(AllocationRequest) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);
  offset = detail::align_up<AllocationState>(offset);
  offset += sizeof(AllocationState);
  offset = detail::align_up<std::max_align_t>(offset);
  offset += algo::cuda::scan::required_workspace_size<std::uint32_t>(
      request_capacity);
  return offset;
}

inline TerminalCollapseWorkspace
create_terminal_collapse_workspace(void *workspace,
                                   std::uint32_t *request_keys,
                                   std::uint32_t request_capacity) {
  TerminalCollapseWorkspace result{};
  result.request_keys = request_keys;
  std::size_t offset = 0;
  reserve_array(workspace, request_capacity, offset, result.unique_offsets);
  reserve_array(workspace, request_capacity, offset, result.unique_requests);
  reserve_array(workspace, 1u, offset, result.unique_count);
  reserve_array(workspace, 1u, offset, result.allocation_state);
  result.scan_workspace_size =
      algo::cuda::scan::required_workspace_size<std::uint32_t>(
          request_capacity);
  reserve_workspace<std::max_align_t>(workspace, result.scan_workspace_size,
                                      offset, result.scan_workspace);
  return result;
}

} // namespace algo::svt::cuda::detail
