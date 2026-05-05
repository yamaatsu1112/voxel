#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/allocation/batch_kernels.cuh>
#include <algo/svt/cuda/detail/allocation/request_compaction.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/plain_depthwise/kernels.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/plain_depthwise/policy.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::plain_depthwise {

// Plain depthwise allocation materializes one tree level at a time. At each
// depth it collects missing parent/child slots, compacts duplicate requests,
// and allocates that batch before moving to the next level.
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

inline cudaError_t allocate_depth(DeviceGpuSvo svo, const LeafMask *leaf_masks,
                                  const std::uint32_t *leaf_count,
                                  std::uint32_t leaf_mask_capacity,
                                  std::uint32_t depth, Workspace &workspace,
                                  cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
        return cudaSuccess;

    const std::uint32_t blocks =
        block_count(leaf_mask_capacity, kKernelBlockSize);
    collect_allocation_requests_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
        svo, leaf_masks, leaf_count, depth, workspace.request_keys,
        leaf_mask_capacity);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    const CompactionInput input{workspace.request_keys,
                                workspace.unique_requests,
                                workspace.unique_count, leaf_mask_capacity};
    status = algo::cuda::scan::exclusive_sum_fused<Transform, PostScan>(
        input, leaf_mask_capacity, workspace.request_scan_workspace,
        workspace.request_scan_workspace_size, stream);
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

} // namespace allocation::plain_depthwise

inline cudaError_t allocation_impl<PlainDepthwiseAllocation>::allocate_paths(
    DeviceGpuSvo svo, const LeafMask *leaf_masks,
    const std::uint32_t *leaf_count, std::uint32_t leaf_mask_capacity,
    Workspace &workspace, cudaStream_t stream) {
    for (std::uint32_t depth = 1u; depth <= kMaxDepth; ++depth) {
        const cudaError_t status = allocation::plain_depthwise::allocate_depth(
            svo, leaf_masks, leaf_count, leaf_mask_capacity, depth, workspace,
            stream);
        if (status != cudaSuccess)
            return status;
    }
    return cudaSuccess;
}

} // namespace algo::svt::cuda::detail
