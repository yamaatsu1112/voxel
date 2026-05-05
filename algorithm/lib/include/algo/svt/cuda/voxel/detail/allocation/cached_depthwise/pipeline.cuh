#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/allocation/batch_kernels.cuh>
#include <algo/svt/cuda/detail/allocation/request_compaction.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/cached_depthwise/kernels.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/cached_depthwise/policy.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::cached_depthwise {

// Depthwise allocation variant that carries each leaf's current parent index
// forward between depths. This avoids re-walking the tree from the root for
// every depth after the first one.
inline cudaError_t allocate_cached_depth(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count, std::uint32_t leaf_mask_capacity,
    std::uint32_t depth, Workspace& workspace,
    cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
        return cudaSuccess;

    const std::uint32_t blocks =
        block_count(leaf_mask_capacity, kKernelBlockSize);
    collect_cached_allocation_requests_kernel<<<blocks, kKernelBlockSize, 0,
                                                stream>>>(
        svo, leaf_masks, leaf_count, workspace.parent_indices, depth,
        workspace.request_keys, leaf_mask_capacity);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    mark_unique_request_offsets_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
        workspace.request_keys, workspace.unique_offsets, leaf_mask_capacity);
    status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    status = algo::cuda::scan::exclusive_sum(
        workspace.unique_offsets, leaf_mask_capacity,
        workspace.request_scan_workspace, workspace.request_scan_workspace_size,
        stream);
    if (status != cudaSuccess)
        return status;

    compact_unique_requests_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
        workspace.request_keys, workspace.unique_offsets, leaf_mask_capacity,
        workspace.unique_requests, workspace.unique_count);
    status = last_launch_status();
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
    status = last_launch_status();
    if (status != cudaSuccess || depth == kMaxDepth)
        return status;

    advance_cached_parent_indices_kernel<<<blocks, kKernelBlockSize, 0,
                                           stream>>>(
        svo, leaf_masks, leaf_count, workspace.parent_indices, depth,
        leaf_mask_capacity);
    return last_launch_status();
}

} // namespace allocation::cached_depthwise

inline cudaError_t allocation_impl<CachedDepthwiseAllocation>::allocate_paths(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count, std::uint32_t leaf_mask_capacity,
    Workspace& workspace, cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
        return cudaSuccess;

    const std::uint32_t blocks =
        block_count(leaf_mask_capacity, kKernelBlockSize);
    initialize_cached_parent_indices_kernel<<<blocks, kKernelBlockSize, 0,
                                              stream>>>(
        workspace.parent_indices, leaf_count, leaf_mask_capacity);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    for (std::uint32_t depth = 1u; depth <= kMaxDepth; ++depth) {
        status = allocation::cached_depthwise::allocate_cached_depth(
            svo, leaf_masks, leaf_count, leaf_mask_capacity, depth, workspace,
            stream);
        if (status != cudaSuccess)
            return status;
    }
    return cudaSuccess;
}

} // namespace algo::svt::cuda::detail
