#pragma once

#include <algo/svt/cuda/voxel/detail/pipeline.cuh>
#include <algo/svt/cuda/voxel/detail/workspace/voxel_to_leaf.cuh>
#include <algo/svt/cuda/voxel/edit_config.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda {

inline std::size_t
voxel_edits_to_leaf_masks_workspace_size(std::uint32_t count) {
    return detail::voxel_to_leaf_workspace_size(count);
}

// Convert raw voxel edits into compact per-leaf masks. This is exposed
// separately because callers may want to inspect or reuse the coalesced edit
// representation without mutating an SVO.
inline cudaError_t
voxel_edits_to_leaf_masks(const VoxelEdit *edits, std::uint32_t count,
                          LeafMask *leaf_masks, std::uint32_t *leaf_count,
                          void *workspace, std::size_t workspace_size,
                          cudaStream_t stream = nullptr) {
    if ((edits == nullptr && count != 0u) || leaf_masks == nullptr ||
        leaf_count == nullptr) {
        return cudaErrorInvalidValue;
    }
    const std::size_t required =
        voxel_edits_to_leaf_masks_workspace_size(count);
    if (workspace_size < required || (workspace == nullptr && required != 0u))
        return cudaErrorInvalidValue;

    const detail::VoxelToLeafWorkspace typed_workspace =
        detail::create_voxel_to_leaf_workspace(workspace, count);
    return detail::build_leaf_masks(edits, count, leaf_masks, leaf_count,
                                    typed_workspace, stream);
}

template <class Config = detail::DefaultEditConfig>
inline std::size_t apply_voxel_edits_workspace_size(std::uint32_t count) {
    return detail::edit_impl<Config>::workspace_size(count);
}

namespace detail {

// Full edit entry point:
// 1. coalesce voxel edits into leaf masks,
// 2. allocate any missing paths to touched leaves,
// 3. set or clear leaf bits,
// 4. collapse now-uniform subtrees back into filled/empty node slots.
template <class Config = DefaultEditConfig>
inline cudaError_t
apply_voxel_edits(DeviceGpuSvo svo, const VoxelEdit *edits, std::uint32_t count,
                  EditOp op, void *workspace, std::size_t workspace_size,
                  cudaStream_t stream = nullptr) {
    if ((edits == nullptr && count != 0u) || workspace == nullptr)
        return cudaErrorInvalidValue;
    if (svo.nodes == nullptr || svo.leaves == nullptr ||
        svo.counters == nullptr)
        return cudaErrorInvalidValue;

    const std::size_t required =
        apply_voxel_edits_workspace_size<Config>(count);
    if (workspace_size < required)
        return cudaErrorInvalidValue;

    return edit_impl<Config>::apply(svo, edits, count, op, workspace,
                                    workspace_size, stream);
}

} // namespace detail

template <class Config = detail::DefaultEditConfig>
inline cudaError_t place_voxel_edits(DeviceGpuSvo svo, const VoxelEdit *edits,
                                     std::uint32_t count, void *workspace,
                                     std::size_t workspace_size,
                                     cudaStream_t stream = nullptr) {
    return detail::apply_voxel_edits<Config>(svo, edits, count, EditOp::Place,
                                             workspace, workspace_size, stream);
}

template <class Config = detail::DefaultEditConfig>
inline cudaError_t destroy_voxel_edits(DeviceGpuSvo svo, const VoxelEdit *edits,
                                       std::uint32_t count, void *workspace,
                                       std::size_t workspace_size,
                                       cudaStream_t stream = nullptr) {
    return detail::apply_voxel_edits<Config>(svo, edits, count, EditOp::Destroy,
                                             workspace, workspace_size, stream);
}

} // namespace algo::svt::cuda
