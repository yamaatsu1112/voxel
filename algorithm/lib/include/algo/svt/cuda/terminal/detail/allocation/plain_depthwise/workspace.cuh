#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct TerminalAllocationWorkspace {
  std::uint32_t *request_keys;
  AllocationRequest *unique_requests;
  std::uint32_t *unique_count;
  AllocationState *allocation_state;
  void *scan_workspace;
  std::size_t scan_workspace_size;
};

inline std::size_t
terminal_allocation_workspace_size(std::uint32_t request_capacity,
                                   std::size_t scan_workspace_size) {
  std::size_t offset = 0;
  offset = detail::align_up<AllocationRequest>(offset);
  offset += sizeof(AllocationRequest) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);
  offset = detail::align_up<AllocationState>(offset);
  offset += sizeof(AllocationState);
  offset = detail::align_up<std::max_align_t>(offset);
  offset += scan_workspace_size;
  return offset;
}

inline TerminalAllocationWorkspace
create_terminal_allocation_workspace(void *workspace,
                                     std::uint32_t *request_keys,
                                     std::uint32_t request_capacity,
                                     std::size_t scan_workspace_size) {
  TerminalAllocationWorkspace result{};
  result.request_keys = request_keys;
  std::size_t offset = 0;
  reserve_array(workspace, request_capacity, offset, result.unique_requests);
  reserve_array(workspace, 1u, offset, result.unique_count);
  reserve_array(workspace, 1u, offset, result.allocation_state);
  result.scan_workspace_size = scan_workspace_size;
  reserve_workspace<std::max_align_t>(workspace, result.scan_workspace_size,
                                      offset, result.scan_workspace);
  return result;
}

} // namespace algo::svt::cuda::detail
