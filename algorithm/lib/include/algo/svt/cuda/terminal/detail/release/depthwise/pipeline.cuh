#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/terminal/detail/release/depthwise/policy.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

namespace algo::svt::cuda::detail::release::depthwise {

__device__ inline std::uint32_t
next_emit_count(DeviceGpuSvo svo, TerminalDetachedChild detached,
                std::uint32_t depth) {
  if (!terminal_release_child_is_active(detached.child_data))
    return 0u;
  if (detached.child_depth > depth)
    return 1u;
  if (detached.child_depth != depth || depth >= kMaxDepth)
    return 0u;

  const std::uint32_t node_index = detached.child_data & kChildIndexMask;
  const GpuSvoNode node = svo.nodes[node_index];
  std::uint32_t count = 0u;
  for (std::uint32_t child = 0u; child < kGroupSize; ++child) {
    if (terminal_release_child_is_active(node.child_data[child]))
      ++count;
  }
  return count;
}

__device__ inline std::uint32_t free_count(TerminalDetachedChild detached,
                                           std::uint32_t depth) {
  if (!terminal_release_child_is_active(detached.child_data))
    return 0u;
  return detached.child_depth == depth ? 1u : 0u;
}

__global__ void mark_counts_kernel(
    DeviceGpuSvo svo, const TerminalDetachedChild *release_frontier,
    const std::uint32_t *release_frontier_count,
    std::uint32_t release_capacity, std::uint32_t depth,
    std::uint32_t *emit_counts, std::uint32_t *free_counts) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= release_capacity)
    return;

  const std::uint32_t count = *release_frontier_count;
  if (index >= count) {
    emit_counts[index] = 0u;
    free_counts[index] = 0u;
    return;
  }

  const TerminalDetachedChild detached = release_frontier[index];
  emit_counts[index] = next_emit_count(svo, detached, depth);
  free_counts[index] = free_count(detached, depth);
}

__global__ void emit_frontier_kernel(
    DeviceGpuSvo svo, const TerminalDetachedChild *release_frontier,
    const std::uint32_t *release_frontier_count,
    TerminalDetachedChild *release_next_frontier,
    std::uint32_t *release_next_frontier_count, std::uint32_t active_count,
    std::uint32_t release_capacity, std::uint32_t depth,
    const std::uint32_t *emit_counts, const std::uint32_t *free_counts,
    const AllocationState *state, cudaError_t *release_status) {
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
    const std::uint32_t freed_count = free_counts[index];
    *release_next_frontier_count = next_count;
    if (next_count > release_capacity) {
      *release_status = cudaErrorInvalidValue;
      return;
    }
    if (depth >= kMaxDepth) {
      if (state->base_free_leaf_count + freed_count > svo.max_leaf_count) {
        *release_status = cudaErrorInvalidValue;
        return;
      }
      svo.counters->free_leaf_count =
          state->base_free_leaf_count + freed_count;
    } else {
      if (state->base_free_node_count + freed_count > svo.max_node_count) {
        *release_status = cudaErrorInvalidValue;
        return;
      }
      svo.counters->free_node_count =
          state->base_free_node_count + freed_count;
    }
  }

  if (index >= count)
    return;

  const TerminalDetachedChild detached = release_frontier[index];
  if (!terminal_release_child_is_active(detached.child_data))
    return;

  const std::uint32_t emit_begin = terminal_prefix_value(emit_counts, index);
  const std::uint32_t free_begin = terminal_prefix_value(free_counts, index);
  const std::uint32_t emit_count = emit_counts[index] - emit_begin;
  const std::uint32_t free_count = free_counts[index] - free_begin;
  if (emit_begin + emit_count > release_capacity) {
    *release_status = cudaErrorInvalidValue;
    return;
  }

  if (detached.child_depth > depth) {
    if (emit_count != 0u)
      release_next_frontier[emit_begin] = detached;
    return;
  }
  if (detached.child_depth != depth)
    return;

  const std::uint32_t storage_index = detached.child_data & kChildIndexMask;
  if (free_count != 0u) {
    if (depth >= kMaxDepth) {
      const std::uint32_t free_index =
          state->base_free_leaf_count + free_begin;
      if (free_index < svo.max_leaf_count)
        svo.free_leaf_indices[free_index] = storage_index;
    } else {
      const std::uint32_t free_index =
          state->base_free_node_count + free_begin;
      if (free_index < svo.max_node_count)
        svo.free_node_indices[free_index] = storage_index;
    }
  }

  if (depth >= kMaxDepth)
    return;

  const GpuSvoNode node = svo.nodes[storage_index];
  std::uint32_t local_emit = 0u;
  for (std::uint32_t child = 0u; child < kGroupSize; ++child) {
    const std::uint32_t child_data = node.child_data[child];
    if (!terminal_release_child_is_active(child_data))
      continue;
    release_next_frontier[emit_begin + local_emit] =
        TerminalDetachedChild{child_data, depth + 1u};
    ++local_emit;
  }
}

inline cudaError_t release(DeviceGpuSvo svo, cudaError_t *status,
                           TerminalDepthwiseReleaseWorkspace workspace,
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

  for (std::uint32_t depth = 1u; depth <= kMaxDepth; ++depth) {
    if (frontier_count > base.capacity) {
      merge_terminal_release_status_kernel<<<1, 1, 0, stream>>>(base.status,
                                                                 status);
      return last_launch_status();
    }
    const std::uint32_t active_count = frontier_count;
    if (active_count == 0u)
      break;
    const std::uint32_t active_blocks =
        block_count(active_count, kKernelBlockSize);
    mark_counts_kernel<<<active_blocks, kKernelBlockSize, 0, stream>>>(
        svo, current_frontier, current_count, active_count, depth,
        base.emit_counts, workspace.free_counts);
    cuda_status = last_launch_status();
    if (cuda_status != cudaSuccess)
      return cuda_status;

    cuda_status = algo::cuda::scan::inclusive_sum(
        base.emit_counts, active_count, base.scan_workspace,
        base.scan_workspace_size, stream);
    if (cuda_status != cudaSuccess)
      return cuda_status;

    cuda_status = algo::cuda::scan::inclusive_sum(
        workspace.free_counts, active_count, base.scan_workspace,
        base.scan_workspace_size, stream);
    if (cuda_status != cudaSuccess)
      return cuda_status;

    snapshot_terminal_release_state_kernel<<<1, 1, 0, stream>>>(
        svo, base.allocation_state);
    cuda_status = last_launch_status();
    if (cuda_status != cudaSuccess)
      return cuda_status;

    emit_frontier_kernel<<<active_blocks, kKernelBlockSize, 0, stream>>>(
        svo, current_frontier, current_count, next_frontier, next_count,
        active_count, base.capacity, depth, base.emit_counts,
        workspace.free_counts, base.allocation_state, base.status);
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

} // namespace algo::svt::cuda::detail::release::depthwise

namespace algo::svt::cuda::detail {

inline cudaError_t terminal_release_impl<TerminalDepthwiseRelease>::release(
    DeviceGpuSvo svo, cudaError_t *status, Workspace workspace,
    cudaStream_t stream) {
  return release::depthwise::release(svo, status, workspace, stream);
}

} // namespace algo::svt::cuda::detail
