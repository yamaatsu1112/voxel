#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/allocation/request_compaction.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/scan_depthwise/kernels.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/scan_depthwise/policy.cuh>

#include <algorithm>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace algo::svt::cuda::detail {

namespace allocation::scan_depthwise {

// Scan-depthwise allocation first records, for each target leaf, where its
// existing path stops. Each depth then emits only the materialization requests
// that are actually needed at that level.
struct Input {
    DeviceGpuSvo svo;
    const LeafMask *leaf_masks;
    const std::uint32_t *leaf_count;
    std::uint32_t leaf_mask_capacity;
    const std::uint32_t *missing_states;
    const std::uint32_t *existing_parent_indices;
    const std::uint32_t *previous_request_emits;
    const AllocationState *previous_allocation_state;
    std::uint32_t *current_request_emits;
    AllocationRequest *unique_requests;
    std::uint32_t *unique_count;
    std::uint32_t depth;
};

struct Transform {
    __device__ static std::uint32_t run(std::uint32_t index,
                                        const Input &input) {
        const std::uint32_t leaf_count = *input.leaf_count;
        if (index >= leaf_count)
            return 0u;

        const std::uint32_t leaf_key = input.leaf_masks[index].leaf_key;
        const bool has_previous_leaf = index != 0u;
        const std::uint32_t previous_leaf_key =
            has_previous_leaf ? input.leaf_masks[index - 1u].leaf_key : 0u;
        const std::uint32_t first_missing_depth =
            missing_state_first_missing_depth(input.missing_states[index]);
        return materialize_slot_emits(leaf_key, previous_leaf_key,
                                      has_previous_leaf, first_missing_depth,
                                      input.depth)
                   ? 1u
                   : 0u;
    }
};

struct PostScan {
    __device__ static void run(std::uint32_t index, const Input &input,
                               std::uint32_t transformed_value,
                               std::uint32_t prefix) {
        input.current_request_emits[index] = prefix;

        if (index == input.leaf_mask_capacity - 1u)
            *input.unique_count = prefix;

        const std::uint32_t leaf_count = *input.leaf_count;
        if (index >= leaf_count || transformed_value == 0u)
            return;

        const std::uint32_t leaf_key = input.leaf_masks[index].leaf_key;
        const std::uint32_t first_missing_depth =
            missing_state_first_missing_depth(input.missing_states[index]);

        std::uint32_t parent_index = input.existing_parent_indices[index];
        if (input.depth > first_missing_depth) {
            const std::uint32_t parent_rank =
                input.previous_request_emits[index];
            parent_index = materialized_node_index(
                input.svo, input.previous_allocation_state, parent_rank - 1u);
        }

        input.unique_requests[prefix - 1u] = AllocationRequest{
            parent_index, child_index_for_leaf_key(leaf_key, input.depth - 1u)};
    }
};

inline cudaError_t allocate_depth(DeviceGpuSvo svo, const LeafMask *leaf_masks,
                                  const std::uint32_t *leaf_count,
                                  std::uint32_t leaf_mask_capacity,
                                  std::uint32_t depth, Workspace &workspace,
                                  cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
        return cudaSuccess;

    snapshot_allocation_state_kernel<<<1, 1, 0, stream>>>(
        svo, workspace.current_allocation_state);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    const Input input{
        svo,
        leaf_masks,
        leaf_count,
        leaf_mask_capacity,
        workspace.missing_states,
        workspace.existing_parent_indices,
        workspace.previous_request_emits,
        workspace.previous_allocation_state,
        workspace.current_request_emits,
        workspace.unique_requests,
        workspace.unique_count,
        depth,
    };
    status = algo::cuda::scan::inclusive_sum_fused<Transform, PostScan>(
        input, leaf_mask_capacity, workspace.request_scan_workspace,
        workspace.request_scan_workspace_size, stream);
    if (status != cudaSuccess)
        return status;

    const std::uint32_t blocks =
        block_count(leaf_mask_capacity, kKernelBlockSize);
    allocate_scan_depth_batch_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
        svo, workspace.unique_requests, workspace.unique_count,
        workspace.current_allocation_state, depth);
    return last_launch_status();
}

} // namespace allocation::scan_depthwise

inline cudaError_t allocation_impl<ScanDepthwiseAllocation>::allocate_paths(
    DeviceGpuSvo svo, const LeafMask *leaf_masks,
    const std::uint32_t *leaf_count, std::uint32_t leaf_mask_capacity,
    Workspace &workspace, cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
        return cudaSuccess;

    const std::uint32_t blocks =
        block_count(leaf_mask_capacity, kKernelBlockSize);
    collect_missing_state_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
        svo, leaf_masks, leaf_count, leaf_mask_capacity,
        workspace.missing_states, workspace.existing_parent_indices);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    for (std::uint32_t depth = 1u; depth <= kMaxDepth; ++depth) {
        status = allocation::scan_depthwise::allocate_depth(
            svo, leaf_masks, leaf_count, leaf_mask_capacity, depth, workspace,
            stream);
        if (status != cudaSuccess)
            return status;
        std::swap(workspace.previous_request_emits,
                  workspace.current_request_emits);
        std::swap(workspace.previous_allocation_state,
                  workspace.current_allocation_state);
    }
    return cudaSuccess;
}

} // namespace algo::svt::cuda::detail
