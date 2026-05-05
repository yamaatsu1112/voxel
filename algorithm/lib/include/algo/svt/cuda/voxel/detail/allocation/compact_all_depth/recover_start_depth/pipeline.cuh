#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/kernels.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/materialize/materializer.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/recover_start_depth/policy.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/recover_start_depth/traits.cuh>
#include <algo/svt/cuda/detail/allocation/request_compaction.cuh>
#include <algo/svt/cuda/detail/launch.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth::recover_start_depth {

// Compact all-depth allocation stores only per-leaf counts and offsets instead
// of an explicit request record for every missing level. InitMode decides how
// materialized nodes and leaves are initialized from those compact offsets.
template <class InitMode, class ScheduleMode>
inline cudaError_t allocate_paths(DeviceGpuSvo svo, const LeafMask* leaf_masks,
                                  const std::uint32_t* leaf_count,
                                  std::uint32_t leaf_mask_capacity,
                                  Workspace& workspace, cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
        return cudaSuccess;

    const std::uint32_t blocks =
        block_count(leaf_mask_capacity, kKernelBlockSize);
    compact_all_depth::collect_materialize_counts_kernel<RecoverStartDepth>
        <<<blocks, kKernelBlockSize, 0, stream>>>(
            svo, leaf_masks, leaf_count, leaf_mask_capacity, workspace);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    status = algo::cuda::scan::inclusive_sum(
        workspace.node_offsets, leaf_mask_capacity,
        workspace.request_scan_workspace, workspace.request_scan_workspace_size,
        stream);
    if (status != cudaSuccess)
        return status;

    status = algo::cuda::scan::inclusive_sum(
        workspace.leaf_offsets, leaf_mask_capacity,
        workspace.request_scan_workspace, workspace.request_scan_workspace_size,
        stream);
    if (status != cudaSuccess)
        return status;

    snapshot_allocation_state_kernel<<<1, 1, 0, stream>>>(
        svo, workspace.allocation_state);
    status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    status =
        compact_all_depth::Materializer<ScheduleMode, InitMode,
                                        RecoverStartDepth>::run(
            svo, leaf_masks, leaf_count, leaf_mask_capacity, workspace, stream);
    if (status != cudaSuccess)
        return status;

    compact_all_depth::
        commit_materialized_allocation_state_kernel<<<1, 1, 0, stream>>>(
            svo, workspace.node_offsets, workspace.leaf_offsets,
            leaf_mask_capacity, workspace.allocation_state);
    return last_launch_status();
}

} // namespace allocation::compact_all_depth::recover_start_depth

template <class InitMode, class ScheduleMode>
inline cudaError_t
allocation_impl<CompactAllDepthAllocation<InitMode, RecoverStartDepth,
                                          ScheduleMode>>::
    allocate_paths(DeviceGpuSvo svo, const LeafMask* leaf_masks,
                   const std::uint32_t* leaf_count,
                   std::uint32_t leaf_mask_capacity, Workspace& workspace,
                   cudaStream_t stream) {
    return allocation::compact_all_depth::recover_start_depth::allocate_paths<
        InitMode, ScheduleMode>(svo, leaf_masks, leaf_count,
                                leaf_mask_capacity, workspace, stream);
}

} // namespace algo::svt::cuda::detail
