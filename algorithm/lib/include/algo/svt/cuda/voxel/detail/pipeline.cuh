#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/edit_common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/all_depth/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/cached_depthwise/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/recover_start_depth/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/store_start_depth/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/plain_depthwise/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/scan_depthwise/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/apply_leaf_masks.cuh>
#include <algo/svt/cuda/voxel/detail/build_leaf_masks/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/collapse/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/dispatch_capacity.cuh>
#include <algo/svt/cuda/voxel/detail/workspace/edit.cuh>
#include <algo/svt/cuda/voxel/edit_config.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

using DefaultEditConfig =
    EditConfig<CompactAllDepthAllocation<Threadwise>, HostLeafCountDispatch>;

template <class Config> struct edit_impl {
    static_assert(kAlwaysFalse<Config>,
                  "edit_impl is not implemented for this edit config");
};

// Shared orchestration for all edit policies. The template parameters select
// the concrete allocation strategy, dispatch capacity rule, collapse strategy,
// and voxel-to-leaf-mask builder.
template <class Allocation, class Dispatch>
inline cudaError_t
apply_edit_pipeline(DeviceGpuSvo svo, const VoxelEdit *edits,
                    std::uint32_t count, EditOp op,
                    EditWorkspace &workspace, cudaStream_t stream) {
    // Raw edits may contain many voxels in the same leaf; collapse them into a
    // bounded list of leaf masks before touching the tree topology.
    cudaError_t status =
        build_leaf_masks(edits, count, workspace.shared.leaf_masks,
                         workspace.shared.leaf_count,
                         workspace.shared.voxel_workspace, stream);
    if (status != cudaSuccess)
        return status;

    if (count == 0u)
        return cudaSuccess;

    std::uint32_t leaf_mask_capacity = 0u;
    status = resolve_dispatch_capacity<Dispatch>(
        workspace.shared.leaf_count, count, &leaf_mask_capacity, stream);
    if (status != cudaSuccess)
        return status;

    // Ensure every touched leaf has a materialized path before applying bits.
    typename allocation_impl<Allocation>::Workspace allocation_workspace =
        allocation_impl<Allocation>::create_workspace(workspace.phase_scratch,
                                                      count);
    status = allocation_impl<Allocation>::allocate_paths(
        svo, workspace.shared.leaf_masks, workspace.shared.leaf_count,
        leaf_mask_capacity, allocation_workspace, stream);
    if (status != cudaSuccess)
        return status;

    status = apply_leaf_masks(svo, workspace.shared.leaf_masks,
                              workspace.shared.leaf_count, leaf_mask_capacity,
                              op, stream);
    if (status != cudaSuccess)
        return status;

    // Destroy edits can leave empty leaves or uniform branches behind. Collapse
    // also gives place edits a chance to represent fully-filled regions
    // compactly.
    collapse::Workspace collapse_workspace =
        create_collapse_workspace(workspace.phase_scratch, count);
    return collapse_uniform_paths(svo, workspace.shared.leaf_masks,
                                  workspace.shared.leaf_count,
                                  leaf_mask_capacity, collapse_workspace,
                                  stream);
}

template <class Allocation, class Dispatch>
struct edit_impl<EditConfig<Allocation, Dispatch>> {
    static std::size_t workspace_size(std::uint32_t count) {
        return edit_workspace_size<Allocation>(count);
    }

    static cudaError_t apply(DeviceGpuSvo svo, const VoxelEdit *edits,
                             std::uint32_t count, EditOp op, void *workspace,
                             std::size_t workspace_size_bytes,
                             cudaStream_t stream) {
        const std::size_t required = workspace_size(count);
        if (workspace_size_bytes < required)
            return cudaErrorInvalidValue;

        EditWorkspace typed_workspace =
            create_edit_workspace<Allocation>(workspace, count);
        return apply_edit_pipeline<Allocation, Dispatch>(
            svo, edits, count, op, typed_workspace, stream);
    }
};

} // namespace algo::svt::cuda::detail
