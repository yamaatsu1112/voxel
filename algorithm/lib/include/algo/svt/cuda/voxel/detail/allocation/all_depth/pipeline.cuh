#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/all_depth/kernels.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/all_depth/policy.cuh>
#include <algo/svt/cuda/detail/allocation/request_compaction.cuh>
#include <algo/svt/cuda/detail/launch.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

// All-depth allocation gathers every missing node/leaf request in one pass, then
// initializes and links all materialized storage from a single allocation-state
// snapshot.
inline cudaError_t allocation_impl<AllDepthAllocation>::allocate_paths(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count, std::uint32_t leaf_mask_capacity,
    Workspace& workspace, cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
        return cudaSuccess;

    const std::uint32_t blocks =
        block_count(leaf_mask_capacity, kKernelBlockSize);
    allocation::all_depth::collect_materialize_flags_kernel<<<
        blocks, kKernelBlockSize, 0, stream>>>(
        svo, leaf_masks, leaf_count, leaf_mask_capacity, workspace.missing_states,
        workspace.existing_parent_indices, workspace.materialize_offsets);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    const std::uint32_t candidate_count =
        allocation::all_depth::materialize_candidate_count(leaf_mask_capacity);
    status = algo::cuda::scan::inclusive_sum(
        workspace.materialize_offsets, candidate_count,
        workspace.request_scan_workspace, workspace.request_scan_workspace_size,
        stream);
    if (status != cudaSuccess)
        return status;

    const std::uint32_t compact_blocks =
        block_count(candidate_count, kKernelBlockSize);
    allocation::all_depth::compact_materialize_requests_kernel<<<
        compact_blocks, kKernelBlockSize, 0, stream>>>(
        leaf_masks, leaf_count, leaf_mask_capacity,
        workspace.missing_states, workspace.existing_parent_indices,
        workspace.materialize_offsets,
        workspace.materialize_requests, workspace.materialize_count,
        workspace.materialize_node_count);
    status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    snapshot_allocation_state_kernel<<<1, 1, 0, stream>>>(
        svo, workspace.allocation_state);
    status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    allocation::all_depth::initialize_materialized_storage_kernel<<<
        compact_blocks, kKernelBlockSize, 0, stream>>>(
        svo, workspace.materialize_requests, workspace.materialize_count,
        workspace.materialize_node_count, workspace.allocation_state);
    status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    allocation::all_depth::link_materialized_storage_kernel<<<
        compact_blocks, kKernelBlockSize, 0, stream>>>(
        svo, workspace.materialize_requests, workspace.materialize_count,
        workspace.materialize_node_count, workspace.allocation_state);
    status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    allocation::all_depth::commit_materialized_allocation_state_kernel<<<
        1, 1, 0, stream>>>(
        svo, workspace.materialize_node_count, workspace.materialize_count,
        workspace.allocation_state);
    return last_launch_status();
}

} // namespace algo::svt::cuda::detail
