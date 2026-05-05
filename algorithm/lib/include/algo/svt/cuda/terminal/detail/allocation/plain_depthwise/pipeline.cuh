#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/allocation/batch_kernels.cuh>
#include <algo/svt/cuda/detail/allocation/request_compaction.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/plain_depthwise/policy.cuh>
#include <algo/svt/cuda/terminal/detail/common.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {
namespace allocation::terminal_plain_depthwise {

struct CompactionInput {
  const std::uint32_t *request_keys;
  AllocationRequest *unique_requests;
  std::uint32_t *unique_count;
  std::uint32_t source_count;
};

struct Transform {
  __device__ static std::uint32_t run(std::uint32_t index,
                                      const CompactionInput &input) {
    const std::uint32_t key = input.request_keys[index];
    if (key == kInvalidRequestKey)
      return 0u;

    const bool unique =
        index == 0u || input.request_keys[index - 1u] != key;
    return unique ? 1u : 0u;
  }
};

struct PostScan {
  __device__ static void run(std::uint32_t index,
                             const CompactionInput &input,
                             std::uint32_t transformed_value,
                             std::uint32_t prefix) {
    if (transformed_value != 0u) {
      const std::uint32_t request_key = input.request_keys[index];
      input.unique_requests[prefix] =
          AllocationRequest{request_key_node_index(request_key),
                            request_key_child_index(request_key)};
    }

    if (index == input.source_count - 1u)
      *input.unique_count = prefix + transformed_value;
  }
};

__device__ inline std::uint32_t
request_materialized_depth(const TerminalRequest &request) {
  if (terminal_request_is_brick(request))
    return kMaxDepth;
  const CellWriteRequest cell = terminal_request_cell(request);
  if (cell.level == 0u)
    return 0u;
  return cell.level - 1u;
}

__global__ void collect_requests_kernel(DeviceGpuSvo svo,
                                        const TerminalRequest *requests,
                                        std::uint32_t request_count,
                                        std::uint32_t depth,
                                        std::uint32_t *request_keys) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= request_count)
    return;

  const TerminalRequest request = requests[index];
  if (depth > request_materialized_depth(request)) {
    request_keys[index] = kInvalidRequestKey;
    return;
  }

  const std::uint32_t leaf_key =
      terminal_request_representative_leaf_key(request);
  std::uint32_t node_index = kRootNodeIndex;
  for (std::uint32_t d = 0u; d + 1u < depth; ++d) {
    const std::uint32_t child_index = child_index_for_leaf_key(leaf_key, d);
    const GpuSvoNode node = svo.nodes[node_index];
    if (!node_has_child(node, child_index)) {
      request_keys[index] = kInvalidRequestKey;
      return;
    }
    node_index = node_child_index(node, child_index);
  }

  const std::uint32_t child_index =
      child_index_for_leaf_key(leaf_key, depth - 1u);
  const GpuSvoNode node = svo.nodes[node_index];
  if (node_has_child(node, child_index)) {
    request_keys[index] = kInvalidRequestKey;
    return;
  }

  request_keys[index] = request_key_for_child(node_index, child_index);
}

inline cudaError_t
allocate_depth(DeviceGpuSvo svo, const TerminalRequest *requests,
               std::uint32_t request_count, std::uint32_t depth,
               TerminalAllocationWorkspace &workspace, cudaStream_t stream) {
  const std::uint32_t blocks = block_count(request_count, kKernelBlockSize);
  collect_requests_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      svo, requests, request_count, depth, workspace.request_keys);
  cudaError_t status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  const CompactionInput input{workspace.request_keys, workspace.unique_requests,
                              workspace.unique_count, request_count};
  status = algo::cuda::scan::exclusive_sum_fused<Transform, PostScan>(
      input, request_count, workspace.scan_workspace,
      workspace.scan_workspace_size, stream);
  if (status != cudaSuccess)
    return status;

  snapshot_allocation_state_kernel<<<1, 1, 0, stream>>>(
      svo, workspace.allocation_state);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  allocate_batch_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      svo, workspace.unique_requests, workspace.unique_count,
      workspace.allocation_state, depth);
  return last_launch_status();
}

inline cudaError_t allocate_paths(DeviceGpuSvo svo,
                                  const TerminalRequest *requests,
                                  std::uint32_t request_count,
                                  TerminalAllocationWorkspace &workspace,
                                  cudaStream_t stream) {
  for (std::uint32_t depth = 1u; depth <= kMaxDepth; ++depth) {
    const cudaError_t status =
        allocate_depth(svo, requests, request_count, depth, workspace, stream);
    if (status != cudaSuccess)
      return status;
  }
  return cudaSuccess;
}

} // namespace allocation::terminal_plain_depthwise

inline cudaError_t
terminal_allocation_impl<TerminalPlainDepthwiseAllocation>::allocate_paths(
    DeviceGpuSvo svo, const TerminalRequest *requests,
    std::uint32_t request_count, Workspace &workspace, cudaStream_t stream) {
  return allocation::terminal_plain_depthwise::allocate_paths(
      svo, requests, request_count, workspace, stream);
}

} // namespace algo::svt::cuda::detail
