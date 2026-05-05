#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/workspace/edit_batch.cuh>
#include <algo/svt/hash_dag_gpu/edit_config.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

template <class> inline constexpr bool kRequestBuilderAlwaysFalse = false;

template <class RequestBuilder> struct request_builder_impl {
  static_assert(kRequestBuilderAlwaysFalse<RequestBuilder>,
                "request_builder_impl is not implemented for this policy");
};

__device__ inline bool request_candidate_emits(const LeafEditMask *leaf_masks,
                                               std::uint32_t leaf_count,
                                               std::uint32_t leaf_index,
                                               std::uint32_t depth) {
  // For each depth, emit the first leaf whose prefix differs from the previous
  // leaf. Because leaf_masks is sorted by leaf_key, this yields one request
  // per distinct touched node at that depth.
  if (leaf_index >= leaf_count)
    return false;
  if (depth == 0u)
    return leaf_index == 0u;

  const std::uint32_t prefix =
      prefix_for_leaf_key(leaf_masks[leaf_index].leaf_key, depth);
  if (leaf_index == 0u)
    return true;
  const std::uint32_t previous_prefix =
      prefix_for_leaf_key(leaf_masks[leaf_index - 1u].leaf_key, depth);
  return prefix != previous_prefix;
}

__global__ void begin_scan_depthwise_request_depth_kernel(
    const std::uint32_t *request_count, std::uint32_t depth,
    std::uint32_t *level_offsets) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  level_offsets[depth] = *request_count;
}

struct FusedScanDepthwiseRequestInput {
  const LeafEditMask *leaf_masks;
  const std::uint32_t *leaf_count;
  std::uint32_t leaf_mask_capacity;
  std::uint32_t depth;
  std::uint32_t *request_offsets;
  EditSvoRequest *requests;
  std::uint32_t *request_count;
  std::uint32_t *level_offsets;
  std::uint32_t *level_counts;
};

struct FusedScanDepthwiseRequestTransform {
  __device__ static std::uint32_t
  run(std::uint32_t index, const FusedScanDepthwiseRequestInput &input) {
    return request_candidate_emits(input.leaf_masks, *input.leaf_count, index,
                                   input.depth)
               ? 1u
               : 0u;
  }
};

struct FusedScanDepthwiseRequestPostScan {
  __device__ static void
  run(std::uint32_t index, const FusedScanDepthwiseRequestInput &input,
      std::uint32_t transformed_value, std::uint32_t prefix) {
    input.request_offsets[index] = prefix;

    if (transformed_value != 0u) {
      const std::uint32_t output = input.level_offsets[input.depth] + prefix - 1u;
      const std::uint32_t leaf_key = input.leaf_masks[index].leaf_key;
      const std::uint32_t request_prefix =
          prefix_for_leaf_key(leaf_key, input.depth);
      input.requests[output] =
          EditSvoRequest{input.depth, request_prefix, kInvalidRequest, 0u,
                         index};
    }

    if (index == input.leaf_mask_capacity - 1u) {
      input.level_counts[input.depth] = prefix;
      *input.request_count = input.level_offsets[input.depth] + prefix;
      if (input.depth + 1u == kEditDepthCount)
        input.level_offsets[kEditDepthCount] = *input.request_count;
    }
  }
};

inline cudaError_t build_scan_depthwise_request_depth(
    std::uint32_t leaf_mask_capacity, std::uint32_t depth,
    EditWorkspace &workspace, cudaStream_t stream) {
  begin_scan_depthwise_request_depth_kernel<<<1, 1, 0, stream>>>(
      workspace.request_count, depth, workspace.level_offsets);
  cudaError_t status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  const FusedScanDepthwiseRequestInput input{
      workspace.leaf_masks,
      workspace.leaf_count,
      leaf_mask_capacity,
      depth,
      workspace.request_offsets,
      workspace.requests,
      workspace.request_count,
      workspace.level_offsets,
      workspace.level_counts,
  };
  return algo::cuda::scan::inclusive_sum_fused<
      FusedScanDepthwiseRequestTransform, FusedScanDepthwiseRequestPostScan>(
      input, leaf_mask_capacity, workspace.scan_workspace,
      workspace.scan_workspace_size, stream);
}

__global__ void link_parent_requests_kernel(EditSvoRequest *requests,
                                            const std::uint32_t *request_count,
                                            const std::uint32_t *level_offsets,
                                            std::uint32_t *edit_status) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= *request_count)
    return;

  EditSvoRequest request = requests[index];
  const std::uint32_t depth = request_depth(request);
  if (depth == 0u) {
    request.parent_request = kInvalidRequest;
    requests[index] = request;
    return;
  }

  const std::uint32_t parent_prefix = request.prefix >> kGroupSizeExp;
  // Requests are ordered by depth and prefix. Binary search in the previous
  // depth range finds the unique parent request for this prefix.
  std::uint32_t left = level_offsets[depth - 1u];
  std::uint32_t right = level_offsets[depth];
  while (left < right) {
    const std::uint32_t mid = left + (right - left) / 2u;
    const std::uint32_t mid_prefix = requests[mid].prefix;
    if (mid_prefix < parent_prefix)
      left = mid + 1u;
    else
      right = mid;
  }

  if (left >= level_offsets[depth] || requests[left].prefix != parent_prefix) {
    set_edit_status(edit_status, kEditStatusWorkspaceOverflow);
    return;
  }

  request.parent_request = left;
  requests[index] = request;
}

inline cudaError_t link_edit_requests(std::uint32_t count,
                                      EditWorkspace &workspace,
                                      cudaStream_t stream) {
  const std::uint32_t max_requests = max_request_count(count);
  const std::uint32_t request_blocks =
      block_count(max_requests, kKernelBlockSize);
  link_parent_requests_kernel<<<request_blocks, kKernelBlockSize, 0, stream>>>(
      workspace.requests, workspace.request_count, workspace.level_offsets,
      workspace.edit_status);
  return last_launch_status();
}

template <> struct request_builder_impl<ScanDepthwiseRequests<Fused>> {
  static cudaError_t build(std::uint32_t count, std::uint32_t leaf_mask_capacity,
                           EditWorkspace &workspace, cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
      return cudaSuccess;

    for (std::uint32_t depth = 0u; depth < kEditDepthCount; ++depth) {
      cudaError_t status = build_scan_depthwise_request_depth(
          leaf_mask_capacity, depth, workspace, stream);
      if (status != cudaSuccess)
        return status;
    }

    return link_edit_requests(count, workspace, stream);
  }
};

} // namespace algo::svt::hash_dag_gpu::detail
