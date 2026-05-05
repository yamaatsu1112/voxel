#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct TerminalCompactAllocationWorkspace {
  std::uint32_t *materialize_target_depths;
  std::uint32_t *materialize_missing_states;
  std::uint32_t *materialize_parent_indices;
  std::uint32_t *materialize_node_offsets;
  std::uint32_t *materialize_leaf_offsets;
  AllocationState *allocation_state;
  void *scan_workspace;
  std::size_t scan_workspace_size;
};

inline std::size_t
terminal_compact_allocation_workspace_size(std::uint32_t request_capacity) {
  std::size_t offset = 0;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<AllocationState>(offset);
  offset += sizeof(AllocationState);
  offset = detail::align_up<std::max_align_t>(offset);
  offset += algo::cuda::scan::required_workspace_size<std::uint32_t>(
      request_capacity);
  return offset;
}

inline TerminalCompactAllocationWorkspace
create_terminal_compact_allocation_workspace(void *workspace,
                                             std::uint32_t request_capacity) {
  TerminalCompactAllocationWorkspace result{};
  std::size_t offset = 0;
  reserve_array(workspace, request_capacity, offset,
                result.materialize_target_depths);
  reserve_array(workspace, request_capacity, offset,
                result.materialize_missing_states);
  reserve_array(workspace, request_capacity, offset,
                result.materialize_parent_indices);
  reserve_array(workspace, request_capacity, offset,
                result.materialize_node_offsets);
  reserve_array(workspace, request_capacity, offset,
                result.materialize_leaf_offsets);
  reserve_array(workspace, 1u, offset, result.allocation_state);
  result.scan_workspace_size =
      algo::cuda::scan::required_workspace_size<std::uint32_t>(
          request_capacity);
  reserve_workspace<std::max_align_t>(workspace, result.scan_workspace_size,
                                      offset, result.scan_workspace);
  return result;
}

} // namespace algo::svt::cuda::detail
