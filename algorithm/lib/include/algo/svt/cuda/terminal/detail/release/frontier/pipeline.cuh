#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/terminal/detail/release/frontier/policy.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

namespace algo::svt::cuda::detail::release::frontier {

__global__ void mark_counts_kernel(
    DeviceGpuSvo svo, const TerminalDetachedChild *release_frontier,
    const std::uint32_t *release_frontier_count,
    std::uint32_t release_capacity, std::uint32_t *emit_counts,
    std::uint32_t *node_free_counts, std::uint32_t *leaf_free_counts) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= release_capacity)
    return;

  const std::uint32_t count = *release_frontier_count;
  if (index >= count) {
    emit_counts[index] = 0u;
    node_free_counts[index] = 0u;
    leaf_free_counts[index] = 0u;
    return;
  }

  const TerminalDetachedChild detached = release_frontier[index];
  emit_counts[index] = 0u;
  node_free_counts[index] = 0u;
  leaf_free_counts[index] = 0u;
  if (!terminal_release_child_is_active(detached.child_data))
    return;

  if (detached.child_depth >= kMaxDepth) {
    leaf_free_counts[index] = 1u;
    return;
  }

  node_free_counts[index] = 1u;
  const std::uint32_t node_index = detached.child_data & kChildIndexMask;
  const GpuSvoNode node = svo.nodes[node_index];
  std::uint32_t emit_count = 0u;
  for (std::uint32_t child = 0u; child < kGroupSize; ++child) {
    if (terminal_release_child_is_active(node.child_data[child]))
      ++emit_count;
  }
  emit_counts[index] = emit_count;
}

__global__ void emit_kernel(
    DeviceGpuSvo svo, const TerminalDetachedChild *release_frontier,
    const std::uint32_t *release_frontier_count,
    TerminalDetachedChild *release_next_frontier,
    std::uint32_t *release_next_frontier_count, std::uint32_t active_count,
    std::uint32_t release_capacity, const std::uint32_t *emit_counts,
    const std::uint32_t *node_free_counts,
    const std::uint32_t *leaf_free_counts, const AllocationState *state,
    cudaError_t *release_status) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = *release_frontier_count;
  if (index >= active_count)
    return;

  if (*release_status != cudaSuccess) {
    if (index == active_count - 1u)
      *release_next_frontier_count = 0u;
    return;
  }

  if (index == active_count - 1u) {
    const std::uint32_t next_count = emit_counts[index];
    const std::uint32_t freed_node_count = node_free_counts[index];
    const std::uint32_t freed_leaf_count = leaf_free_counts[index];
    *release_next_frontier_count = next_count;
    if (next_count > release_capacity) {
      *release_status = cudaErrorInvalidValue;
      return;
    }
    if (state->base_free_node_count + freed_node_count > svo.max_node_count) {
      *release_status = cudaErrorInvalidValue;
      return;
    }
    if (state->base_free_leaf_count + freed_leaf_count > svo.max_leaf_count) {
      *release_status = cudaErrorInvalidValue;
      return;
    }
    svo.counters->free_node_count =
        state->base_free_node_count + freed_node_count;
    svo.counters->free_leaf_count =
        state->base_free_leaf_count + freed_leaf_count;
  }

  if (index >= count)
    return;

  const TerminalDetachedChild detached = release_frontier[index];
  if (!terminal_release_child_is_active(detached.child_data))
    return;

  const std::uint32_t emit_begin = terminal_prefix_value(emit_counts, index);
  const std::uint32_t node_free_begin =
      terminal_prefix_value(node_free_counts, index);
  const std::uint32_t leaf_free_begin =
      terminal_prefix_value(leaf_free_counts, index);
  const std::uint32_t emit_count = emit_counts[index] - emit_begin;
  const std::uint32_t node_free_count =
      node_free_counts[index] - node_free_begin;
  const std::uint32_t leaf_free_count =
      leaf_free_counts[index] - leaf_free_begin;
  if (emit_begin + emit_count > release_capacity) {
    *release_status = cudaErrorInvalidValue;
    return;
  }

  const std::uint32_t storage_index = detached.child_data & kChildIndexMask;
  if (leaf_free_count != 0u) {
    const std::uint32_t free_index =
        state->base_free_leaf_count + leaf_free_begin;
    if (free_index < svo.max_leaf_count)
      svo.free_leaf_indices[free_index] = storage_index;
    return;
  }

  if (node_free_count == 0u)
    return;

  const std::uint32_t free_index =
      state->base_free_node_count + node_free_begin;
  if (free_index < svo.max_node_count)
    svo.free_node_indices[free_index] = storage_index;

  const GpuSvoNode node = svo.nodes[storage_index];
  std::uint32_t local_emit = 0u;
  for (std::uint32_t child = 0u; child < kGroupSize; ++child) {
    const std::uint32_t child_data = node.child_data[child];
    if (!terminal_release_child_is_active(child_data))
      continue;
    release_next_frontier[emit_begin + local_emit] =
        TerminalDetachedChild{child_data, detached.child_depth + 1u};
    ++local_emit;
  }
}

inline cudaError_t release(DeviceGpuSvo svo, cudaError_t *status,
                           TerminalFrontierReleaseWorkspace workspace,
                           cudaStream_t stream) {
  auto base = workspace.base;
  if (base.capacity == 0u)
    return cudaSuccess;

  cudaError_t cuda_status =
      cudaMemsetAsync(base.status, 0, sizeof(cudaError_t), stream);
  if (cuda_status != cudaSuccess)
    return cuda_status;

  std::uint32_t frontier_count = 0u;
  cuda_status =
      cudaMemcpyAsync(&frontier_count, base.detached_child_count,
                      sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
  if (cuda_status != cudaSuccess)
    return cuda_status;
  cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess)
    return cuda_status;
  if (frontier_count > base.capacity) {
    merge_terminal_release_status_kernel<<<1, 1, 0, stream>>>(base.status,
                                                               status);
    return last_launch_status();
  }

  const std::uint32_t initial_count = frontier_count;
  const std::uint32_t blocks =
      block_count(initial_count == 0u ? 1u : initial_count, kKernelBlockSize);
  initialize_terminal_release_frontier_kernel<<<blocks, kKernelBlockSize, 0,
                                                stream>>>(
      base.detached_children, base.detached_child_count, base.frontier,
      base.frontier_count, base.capacity, base.status);
  cuda_status = last_launch_status();
  if (cuda_status != cudaSuccess)
    return cuda_status;

  TerminalDetachedChild *current_frontier = base.frontier;
  TerminalDetachedChild *next_frontier = base.next_frontier;
  std::uint32_t *current_count = base.frontier_count;
  std::uint32_t *next_count = base.next_frontier_count;

  while (frontier_count != 0u) {
    if (frontier_count > base.capacity) {
      merge_terminal_release_status_kernel<<<1, 1, 0, stream>>>(base.status,
                                                                 status);
      return last_launch_status();
    }

    const std::uint32_t active_count = frontier_count;
    const std::uint32_t active_blocks =
        block_count(active_count, kKernelBlockSize);
    mark_counts_kernel<<<active_blocks, kKernelBlockSize, 0, stream>>>(
        svo, current_frontier, current_count, active_count, base.emit_counts,
        workspace.node_free_counts, workspace.leaf_free_counts);
    cuda_status = last_launch_status();
    if (cuda_status != cudaSuccess)
      return cuda_status;

    cuda_status = algo::cuda::scan::inclusive_sum(
        base.emit_counts, active_count, base.scan_workspace,
        base.scan_workspace_size, stream);
    if (cuda_status != cudaSuccess)
      return cuda_status;

    cuda_status = algo::cuda::scan::inclusive_sum(
        workspace.node_free_counts, active_count, base.scan_workspace,
        base.scan_workspace_size, stream);
    if (cuda_status != cudaSuccess)
      return cuda_status;

    cuda_status = algo::cuda::scan::inclusive_sum(
        workspace.leaf_free_counts, active_count, base.scan_workspace,
        base.scan_workspace_size, stream);
    if (cuda_status != cudaSuccess)
      return cuda_status;

    snapshot_terminal_release_state_kernel<<<1, 1, 0, stream>>>(
        svo, base.allocation_state);
    cuda_status = last_launch_status();
    if (cuda_status != cudaSuccess)
      return cuda_status;

    emit_kernel<<<active_blocks, kKernelBlockSize, 0, stream>>>(
        svo, current_frontier, current_count, next_frontier, next_count,
        active_count, base.capacity, base.emit_counts,
        workspace.node_free_counts, workspace.leaf_free_counts,
        base.allocation_state, base.status);
    cuda_status = last_launch_status();
    if (cuda_status != cudaSuccess)
      return cuda_status;

    cuda_status =
        cudaMemcpyAsync(&frontier_count, next_count, sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream);
    if (cuda_status != cudaSuccess)
      return cuda_status;
    cuda_status = cudaStreamSynchronize(stream);
    if (cuda_status != cudaSuccess)
      return cuda_status;
    std::swap(current_frontier, next_frontier);
    std::swap(current_count, next_count);
  }

  merge_terminal_release_status_kernel<<<1, 1, 0, stream>>>(base.status,
                                                             status);
  return last_launch_status();
}

} // namespace algo::svt::cuda::detail::release::frontier

namespace algo::svt::cuda::detail {

inline cudaError_t terminal_release_impl<TerminalFrontierRelease>::release(
    DeviceGpuSvo svo, cudaError_t *status, Workspace workspace,
    cudaStream_t stream) {
  return release::frontier::release(svo, status, workspace, stream);
}

} // namespace algo::svt::cuda::detail
