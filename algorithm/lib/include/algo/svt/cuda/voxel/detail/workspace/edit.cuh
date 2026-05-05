#pragma once

#include <algo/svt/cuda/detail/workspace_layout.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/common.cuh>
#include <algo/svt/cuda/voxel/detail/collapse/workspace.cuh>
#include <algo/svt/cuda/voxel/detail/workspace/voxel_to_leaf.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct EditSharedWorkspace {
    LeafMask* leaf_masks;
    std::uint32_t* leaf_count;
    VoxelToLeafWorkspace voxel_workspace;
};

struct EditWorkspace {
    EditSharedWorkspace shared;
    void* phase_scratch;
    std::size_t phase_scratch_size;
};

inline std::size_t edit_shared_workspace_size(std::uint32_t count) {
    std::size_t offset = 0;
    offset = align_up<LeafMask>(offset);
    offset += sizeof(LeafMask) * count;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t);
    offset = align_up<std::max_align_t>(offset);
    offset += voxel_to_leaf_workspace_size(count);
    return align_up<std::max_align_t>(offset);
}

template <class Allocation>
inline EditWorkspace create_edit_workspace(void* workspace,
                                           std::uint32_t count) {
    EditWorkspace result{};
    std::size_t offset = 0;
    reserve_array(workspace, count, offset, result.shared.leaf_masks);
    reserve_array(workspace, 1, offset, result.shared.leaf_count);

    offset = align_up<std::max_align_t>(offset);
    void* voxel_workspace = pointer_at<std::byte>(workspace, offset);
    result.shared.voxel_workspace =
        create_voxel_to_leaf_workspace(voxel_workspace, count);

    const std::size_t shared_size = edit_shared_workspace_size(count);
    result.phase_scratch = pointer_at<std::byte>(workspace, shared_size);
    result.phase_scratch_size =
        std::max(allocation_impl<Allocation>::workspace_size(count),
                 collapse_workspace_size(count));
    return result;
}

template <class Allocation>
inline std::size_t edit_workspace_size(std::uint32_t count) {
    return edit_shared_workspace_size(count) +
           std::max(allocation_impl<Allocation>::workspace_size(count),
                    collapse_workspace_size(count));
}

} // namespace algo::svt::cuda::detail
