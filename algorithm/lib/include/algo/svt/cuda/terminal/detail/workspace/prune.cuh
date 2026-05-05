#pragma once

#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct TerminalPruneWorkspace {
  std::uint32_t *coverage_offsets;
  std::uint32_t *emit_offsets;
  void *scan_workspace;
  std::size_t scan_workspace_size;
};

inline std::size_t
terminal_prune_workspace_size(std::uint32_t request_capacity) {
  std::size_t offset = 0;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<std::max_align_t>(offset);
  offset += terminal_common_scan_workspace_size(request_capacity,
                                                request_capacity);
  return offset;
}

inline TerminalPruneWorkspace
create_terminal_prune_workspace(void *workspace,
                                std::uint32_t request_capacity) {
  TerminalPruneWorkspace result{};
  std::size_t offset = 0;
  reserve_array(workspace, request_capacity, offset, result.coverage_offsets);
  reserve_array(workspace, request_capacity, offset, result.emit_offsets);
  result.scan_workspace_size =
      terminal_common_scan_workspace_size(request_capacity, request_capacity);
  reserve_workspace<std::max_align_t>(workspace, result.scan_workspace_size,
                                      offset, result.scan_workspace);
  return result;
}

} // namespace algo::svt::cuda::detail
