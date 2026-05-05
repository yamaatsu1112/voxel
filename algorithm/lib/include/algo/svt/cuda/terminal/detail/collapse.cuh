#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/allocation/batch_kernels.cuh>
#include <algo/svt/cuda/detail/allocation/request_compaction.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/terminal/detail/common.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

__global__ void collect_terminal_free_requests_kernel(
    DeviceGpuSvo svo, const TerminalRequest *requests,
    std::uint32_t request_count, std::uint32_t depth,
    std::uint32_t *request_keys) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= request_count)
    return;

  const std::uint32_t leaf_key =
      terminal_request_representative_leaf_key(requests[index]);
  std::uint32_t node_index = kRootNodeIndex;
  std::uint32_t parent_index = kRootNodeIndex;
  std::uint32_t parent_child_index = 0u;

  if (depth == kMaxDepth) {
    for (std::uint32_t d = 0u; d < kMaxDepth; ++d) {
      const std::uint32_t child_index =
          child_index_for_leaf_key(leaf_key, d);
      const GpuSvoNode node = svo.nodes[node_index];
      if (!node_has_child(node, child_index)) {
        request_keys[index] = kInvalidRequestKey;
        return;
      }
      if (d == kMaxDepth - 1u) {
        parent_index = node_index;
        parent_child_index = child_index;
      }
      node_index = node_child_index(node, child_index);
    }

    const GpuSvoLeaf leaf = svo.leaves[node_index];
    const bool empty =
        leaf.voxel_data_low == 0u && leaf.voxel_data_high == 0u;
    const bool full = leaf.voxel_data_low == 0xffffffffu &&
                      leaf.voxel_data_high == 0xffffffffu;
    if (!empty && !full) {
      request_keys[index] = kInvalidRequestKey;
      return;
    }
  } else {
    for (std::uint32_t d = 0u; d < depth; ++d) {
      const std::uint32_t child_index =
          child_index_for_leaf_key(leaf_key, d);
      const GpuSvoNode node = svo.nodes[node_index];
      if (!node_has_child(node, child_index)) {
        request_keys[index] = kInvalidRequestKey;
        return;
      }
      if (d == depth - 1u) {
        parent_index = node_index;
        parent_child_index = child_index;
      }
      node_index = node_child_index(node, child_index);
    }

    const GpuSvoNode node = svo.nodes[node_index];
    bool all_empty = true;
    bool all_full = true;
    for (std::uint32_t child = 0u; child < kGroupSize; ++child) {
      if (node_has_child(node, child)) {
        all_empty = false;
        all_full = false;
        break;
      }
      if (node_is_filled(node, child)) {
        all_empty = false;
      } else {
        all_full = false;
      }
    }

    if (!all_empty && !all_full) {
      request_keys[index] = kInvalidRequestKey;
      return;
    }
  }

  request_keys[index] = request_key_for_child(parent_index, parent_child_index);
}

inline cudaError_t collapse_terminal_request_depth(
    DeviceGpuSvo svo, const TerminalRequest *requests,
    std::uint32_t request_count, std::uint32_t depth,
    TerminalCollapseWorkspace workspace, cudaStream_t stream) {
  const std::uint32_t blocks = block_count(request_count, kKernelBlockSize);
  collect_terminal_free_requests_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      svo, requests, request_count, depth, workspace.request_keys);
  cudaError_t status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  mark_unique_request_offsets_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.request_keys, workspace.unique_offsets, request_count);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  status = algo::cuda::scan::exclusive_sum(
      workspace.unique_offsets, request_count, workspace.scan_workspace,
      workspace.scan_workspace_size, stream);
  if (status != cudaSuccess)
    return status;

  compact_unique_requests_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.request_keys, workspace.unique_offsets, request_count,
      workspace.unique_requests, workspace.unique_count);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  snapshot_allocation_state_kernel<<<1, 1, 0, stream>>>(
      svo, workspace.allocation_state);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  free_batch_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      svo, workspace.unique_requests, workspace.unique_count,
      workspace.allocation_state, depth);
  return last_launch_status();
}

inline cudaError_t collapse_terminal_requests(
    DeviceGpuSvo svo, const TerminalRequest *requests,
    std::uint32_t request_count, TerminalCollapseWorkspace workspace,
    cudaStream_t stream) {
  for (std::uint32_t depth = kMaxDepth; depth >= 1u; --depth) {
    const cudaError_t status = collapse_terminal_request_depth(
        svo, requests, request_count, depth, workspace, stream);
    if (status != cudaSuccess)
      return status;
  }
  return cudaSuccess;
}

} // namespace algo::svt::cuda::detail
