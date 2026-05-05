#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/workspace_layout.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/common.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

template <class Release> struct terminal_release_impl;

struct TerminalDetachedChild {
  std::uint32_t child_data;
  std::uint32_t child_depth;
};

struct TerminalEditWorkspace {
  std::uint32_t *request_offsets;
  cudaError_t *status;
  TerminalRequest *requests;
  TerminalRequest *pruned_requests;
  std::uint32_t *request_keys;
  TerminalDetachedChild *detached_children;
  std::uint32_t *detached_child_count;
  std::uint32_t request_capacity;
  void *phase_scratch;
  std::size_t phase_scratch_size;
};

inline std::size_t
terminal_common_scan_workspace_size(std::uint32_t count_segment_count,
                                    std::uint32_t request_capacity) {
  return std::max(
      algo::cuda::scan::required_workspace_size<std::uint32_t>(
          std::max(count_segment_count, request_capacity)),
      algo::cuda::scan::required_workspace_size<
          algo::cuda::scan::detail::DefaultConfig,
          algo::cuda::scan::Max<std::uint32_t>, std::uint32_t>(
          request_capacity));
}

inline std::size_t
terminal_edit_workspace_size_with_phase(std::uint32_t node_count,
                                        std::uint32_t leaf_count,
                                        std::uint32_t request_capacity,
                                        std::size_t phase_scratch_size) {
  const std::uint32_t count_segment_count = node_count * 2u + leaf_count;
  std::size_t offset = 0;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * count_segment_count;
  offset = detail::align_up<cudaError_t>(offset);
  offset += sizeof(cudaError_t);
  offset = detail::align_up<TerminalRequest>(offset);
  offset += sizeof(TerminalRequest) * request_capacity;
  offset = detail::align_up<TerminalRequest>(offset);
  offset += sizeof(TerminalRequest) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_capacity;
  offset = detail::align_up<TerminalDetachedChild>(offset);
  offset += sizeof(TerminalDetachedChild) * request_capacity;
  offset = detail::align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);
  offset = detail::align_up<std::max_align_t>(offset);
  offset += phase_scratch_size;
  return offset;
}

inline TerminalEditWorkspace
create_terminal_edit_workspace_with_phase(void *workspace,
                                          std::uint32_t node_count,
                                          std::uint32_t leaf_count,
                                          std::uint32_t request_capacity,
                                          std::size_t phase_scratch_size) {
  TerminalEditWorkspace result{};
  result.request_capacity = request_capacity;
  const std::uint32_t count_segment_count = node_count * 2u + leaf_count;

  std::size_t offset = 0;
  reserve_array(workspace, count_segment_count, offset, result.request_offsets);
  reserve_array(workspace, 1u, offset, result.status);
  reserve_array(workspace, result.request_capacity, offset, result.requests);
  reserve_array(workspace, result.request_capacity, offset,
                result.pruned_requests);
  reserve_array(workspace, result.request_capacity, offset,
                result.request_keys);
  reserve_array(workspace, result.request_capacity, offset,
                result.detached_children);
  reserve_array(workspace, 1u, offset, result.detached_child_count);
  result.phase_scratch_size = phase_scratch_size;
  reserve_workspace<std::max_align_t>(workspace, result.phase_scratch_size,
                                      offset, result.phase_scratch);
  return result;
}

} // namespace algo::svt::cuda::detail

#include <algo/svt/cuda/terminal/detail/workspace/collapse.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/count_sort.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/prune.cuh>
#include <algo/svt/cuda/terminal/detail/release/depthwise/policy.cuh>
#include <algo/svt/cuda/terminal/detail/release/frontier/policy.cuh>

namespace algo::svt::cuda::detail {

template <class Allocation, class Release>
inline std::size_t terminal_phase_scratch_size(
    std::uint32_t count_segment_count, std::uint32_t request_capacity) {
  return std::max({
      terminal_count_sort_workspace_size(count_segment_count, request_capacity),
      terminal_prune_workspace_size(request_capacity),
      terminal_allocation_impl<Allocation>::allocation_workspace_size(
          request_capacity),
      terminal_release_impl<Release>::release_workspace_size(request_capacity),
      terminal_collapse_workspace_size(request_capacity),
  });
}

template <class Allocation, class Release>
inline std::size_t
apply_terminal_edits_workspace_size(std::uint32_t node_count,
                                    std::uint32_t leaf_count,
                                    std::uint32_t request_capacity) {
  const std::uint32_t count_segment_count = node_count * 2u + leaf_count;
  return terminal_edit_workspace_size_with_phase(
      node_count, leaf_count, request_capacity,
      terminal_phase_scratch_size<Allocation, Release>(count_segment_count,
                                                       request_capacity));
}

template <class Allocation, class Release>
inline TerminalEditWorkspace
create_terminal_edit_workspace(void *workspace, std::uint32_t node_count,
                               std::uint32_t leaf_count,
                               std::size_t workspace_size) {
  const std::size_t fixed_size =
      apply_terminal_edits_workspace_size<Allocation, Release>(
          node_count, leaf_count, 0u);
  std::size_t request_capacity =
      workspace_size < fixed_size
          ? 0u
          : (workspace_size - fixed_size) /
                (sizeof(TerminalRequest) * 2u + sizeof(std::uint32_t) +
                 sizeof(TerminalDetachedChild));

  std::size_t low_capacity = 0u;
  std::size_t high_capacity = request_capacity + 1u;
  while (low_capacity + 1u < high_capacity) {
    const std::size_t mid_capacity =
        low_capacity + (high_capacity - low_capacity) / 2u;
    const std::size_t required =
        apply_terminal_edits_workspace_size<Allocation, Release>(
            node_count, leaf_count, static_cast<std::uint32_t>(mid_capacity));
    if (required <= workspace_size) {
      low_capacity = mid_capacity;
    } else {
      high_capacity = mid_capacity;
    }
  }

  const auto capacity = static_cast<std::uint32_t>(low_capacity);
  const std::uint32_t count_segment_count = node_count * 2u + leaf_count;
  return create_terminal_edit_workspace_with_phase(
      workspace, node_count, leaf_count, capacity,
      terminal_phase_scratch_size<Allocation, Release>(count_segment_count,
                                                       capacity));
}

} // namespace algo::svt::cuda::detail
