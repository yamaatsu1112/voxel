#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/cuda/sort/sort.cuh>
#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/edit_common.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/allocation.cuh>
#include <algo/svt/cuda/terminal/detail/apply_writes.cuh>
#include <algo/svt/cuda/terminal/detail/collapse.cuh>
#include <algo/svt/cuda/terminal/detail/count_requests.cuh>
#include <algo/svt/cuda/terminal/detail/emit_requests.cuh>
#include <algo/svt/cuda/terminal/detail/prune.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>
#include <algo/svt/cuda/terminal/edit_config.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

template <class Allocation, class Release>
inline cudaError_t
apply_terminal_edits(DeviceGpuSvo svo, const TerminalNodeInput *nodes,
                     std::uint32_t node_count, const TerminalLeafInput *leaves,
                     std::uint32_t leaf_count, EditOp op, void *workspace,
                     std::size_t workspace_size,
                     cudaStream_t stream = nullptr) {
  if ((nodes == nullptr && node_count != 0u) ||
      (leaves == nullptr && leaf_count != 0u)) {
    return cudaErrorInvalidValue;
  }
  if (svo.nodes == nullptr || svo.leaves == nullptr || svo.counters == nullptr)
    return cudaErrorInvalidValue;

  const std::uint32_t input_count = node_count + leaf_count;
  const std::size_t minimum_required =
      apply_terminal_edits_workspace_size<Allocation, Release>(
          node_count, leaf_count, 0u);
  if (workspace == nullptr || workspace_size < minimum_required)
    return cudaErrorInvalidValue;

  TerminalEditWorkspace typed_workspace =
      create_terminal_edit_workspace<Allocation, Release>(
          workspace, node_count, leaf_count, workspace_size);

  cudaError_t status =
      cudaMemsetAsync(typed_workspace.status, 0, sizeof(cudaError_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(typed_workspace.detached_child_count, 0,
                           sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;

  if (input_count == 0u)
    return cudaSuccess;

  const std::uint32_t count_segment_count = node_count * 2u + leaf_count;
  TerminalCountSortWorkspace count_sort_workspace =
      create_terminal_count_sort_workspace(typed_workspace.phase_scratch,
                                           count_segment_count,
                                           typed_workspace.request_capacity);
  const std::uint32_t count_segment_blocks =
      detail::block_count(count_segment_count, detail::kKernelBlockSize);
  detail::count_terminal_requests_kernel<<<
      count_segment_blocks, detail::kKernelBlockSize, 0, stream>>>(
      nodes, node_count, leaves, leaf_count, typed_workspace.request_offsets);
  status = detail::last_launch_status();
  if (status != cudaSuccess)
    return status;

  status = algo::cuda::scan::inclusive_sum(
      typed_workspace.request_offsets, count_segment_count,
      count_sort_workspace.scan_workspace,
      count_sort_workspace.scan_workspace_size, stream);
  if (status != cudaSuccess)
    return status;

  std::uint32_t request_count = 0u;
  status = cudaMemcpyAsync(
      &request_count,
      typed_workspace.request_offsets + count_segment_count - 1u,
      sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess)
    return status;
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess)
    return status;
  if (request_count > typed_workspace.request_capacity)
    return cudaErrorInvalidValue;
  if (request_count == 0u)
    return cudaSuccess;

  const std::uint32_t emit_request_blocks =
      detail::block_count(request_count, detail::kKernelBlockSize);
  detail::emit_terminal_requests_kernel<<<
      emit_request_blocks, detail::kKernelBlockSize, 0, stream>>>(
      nodes, node_count, leaves, leaf_count, typed_workspace.request_offsets,
      count_segment_count, request_count, typed_workspace.requests,
      typed_workspace.request_keys);
  status = detail::last_launch_status();
  if (status != cudaSuccess)
    return status;

  status = algo::cuda::sort::sort_pairs(
      typed_workspace.request_keys, typed_workspace.requests, request_count,
      count_sort_workspace.sort_workspace,
      count_sort_workspace.sort_workspace_size, stream);
  if (status != cudaSuccess)
    return status;

  status = detail::prune_terminal_requests(
      typed_workspace.requests, typed_workspace.pruned_requests, request_count,
      typed_workspace.request_keys,
      create_terminal_prune_workspace(typed_workspace.phase_scratch,
                                      typed_workspace.request_capacity),
      stream);
  if (status != cudaSuccess)
    return status;
  if (request_count == 0u)
    return cudaSuccess;

  const TerminalRequest *active_requests = typed_workspace.pruned_requests;
  const std::uint32_t request_blocks =
      detail::block_count(request_count, detail::kKernelBlockSize);

  status = detail::allocate_terminal_request_paths<Allocation>(
      svo, active_requests, request_count, typed_workspace.request_keys,
      typed_workspace.phase_scratch, stream);
  if (status != cudaSuccess)
    return status;

  const bool filled = op == EditOp::Place;
  detail::apply_terminal_brick_requests_kernel<<<
      request_blocks, detail::kKernelBlockSize, 0, stream>>>(
      svo, active_requests, request_count, filled, typed_workspace.status);
  status = detail::last_launch_status();
  if (status != cudaSuccess)
    return status;

  detail::apply_terminal_cell_requests_kernel<<<
      request_blocks, detail::kKernelBlockSize, 0, stream>>>(
      svo, active_requests, request_count, filled, typed_workspace.status,
      typed_workspace.detached_children, typed_workspace.detached_child_count,
      typed_workspace.request_capacity);
  status = detail::last_launch_status();
  if (status != cudaSuccess)
    return status;

  status = detail::release_terminal_detached_children<Release>(
      svo, typed_workspace.status,
      terminal_release_impl<Release>::create_phase_workspace(
          typed_workspace.phase_scratch, typed_workspace.detached_children,
          typed_workspace.detached_child_count,
          typed_workspace.request_capacity),
      stream);
  if (status != cudaSuccess)
    return status;

  cudaError_t host_status = cudaSuccess;
  status = cudaMemcpyAsync(&host_status, typed_workspace.status,
                           sizeof(cudaError_t), cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess)
    return status;
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess)
    return status;
  if (host_status != cudaSuccess)
    return host_status;

  status = detail::collapse_terminal_requests(
      svo, active_requests, request_count,
      create_terminal_collapse_workspace(typed_workspace.phase_scratch,
                                         typed_workspace.request_keys,
                                         typed_workspace.request_capacity),
      stream);
  if (status != cudaSuccess)
    return status;

  status = cudaMemcpyAsync(&host_status, typed_workspace.status,
                           sizeof(cudaError_t), cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess)
    return status;
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess)
    return status;
  return host_status;
}

} // namespace algo::svt::cuda::detail

namespace algo::svt::cuda {

template <class Config = DefaultTerminalEditConfig>
inline cudaError_t place_terminal_edits(
    DeviceGpuSvo svo, const TerminalNodeInput *nodes, std::uint32_t node_count,
    const TerminalLeafInput *leaves, std::uint32_t leaf_count, void *workspace,
    std::size_t workspace_size, cudaStream_t stream = nullptr) {
  return detail::apply_terminal_edits<typename Config::allocation,
                                      typename Config::release>(
      svo, nodes, node_count, leaves, leaf_count, EditOp::Place, workspace,
      workspace_size, stream);
}

template <class Config = DefaultTerminalEditConfig>
inline cudaError_t destroy_terminal_edits(
    DeviceGpuSvo svo, const TerminalNodeInput *nodes, std::uint32_t node_count,
    const TerminalLeafInput *leaves, std::uint32_t leaf_count, void *workspace,
    std::size_t workspace_size, cudaStream_t stream = nullptr) {
  return detail::apply_terminal_edits<typename Config::allocation,
                                      typename Config::release>(
      svo, nodes, node_count, leaves, leaf_count, EditOp::Destroy, workspace,
      workspace_size, stream);
}

} // namespace algo::svt::cuda
