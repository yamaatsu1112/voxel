#pragma once

#include <algo/cuda/sort/sort.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct TerminalCountSortWorkspace {
  void *scan_workspace;
  std::size_t scan_workspace_size;
  void *sort_workspace;
  std::size_t sort_workspace_size;
};

inline std::size_t
terminal_count_sort_workspace_size(std::uint32_t count_segment_count,
                                   std::uint32_t request_capacity) {
  std::size_t offset = 0;
  offset = detail::align_up<std::max_align_t>(offset);
  offset += terminal_common_scan_workspace_size(count_segment_count,
                                                request_capacity);
  offset = detail::align_up<std::max_align_t>(offset);
  offset += algo::cuda::sort::required_pairs_workspace_size<std::uint32_t,
                                                            TerminalRequest>(
      request_capacity);
  return offset;
}

inline TerminalCountSortWorkspace
create_terminal_count_sort_workspace(void *workspace,
                                     std::uint32_t count_segment_count,
                                     std::uint32_t request_capacity) {
  TerminalCountSortWorkspace result{};
  std::size_t offset = 0;
  result.scan_workspace_size =
      terminal_common_scan_workspace_size(count_segment_count,
                                          request_capacity);
  reserve_workspace<std::max_align_t>(workspace, result.scan_workspace_size,
                                      offset, result.scan_workspace);
  result.sort_workspace_size = algo::cuda::sort::required_pairs_workspace_size<
      std::uint32_t, TerminalRequest>(request_capacity);
  reserve_workspace<std::max_align_t>(workspace, result.sort_workspace_size,
                                      offset, result.sort_workspace);
  return result;
}

} // namespace algo::svt::cuda::detail
