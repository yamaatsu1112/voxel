#pragma once

#include <algo/cuda/sort/sort.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/voxel/detail/build_leaf_masks/builder.cuh>
#include <algo/svt/cuda/voxel/detail/workspace/voxel_to_leaf.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

// First stage of edit coalescing. Each valid voxel edit becomes a sortable
// (leaf_key, leaf-local bit) pair; invalid edits sort to kInvalidSortKey and
// are dropped by the leaf-mask builder.
__global__ void pack_voxel_edit_records_kernel(const VoxelEdit *edits,
                                               std::uint32_t count,
                                               std::uint32_t *keys,
                                               std::uint32_t *local_offsets) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const VoxelEdit edit = edits[index];
    if (!valid_world_coord(edit.x) || !valid_world_coord(edit.y) ||
        !valid_world_coord(edit.z)) {
        keys[index] = kInvalidSortKey;
        local_offsets[index] = 0u;
        return;
    }

    const std::uint32_t leaf_x = edit.x >> kLeafVoxelCountExp;
    const std::uint32_t leaf_y = edit.y >> kLeafVoxelCountExp;
    const std::uint32_t leaf_z = edit.z >> kLeafVoxelCountExp;
    keys[index] = make_leaf_key(leaf_x, leaf_y, leaf_z);
    local_offsets[index] = leaf_offset_for_voxel(edit.x, edit.y, edit.z);
}

inline cudaError_t build_leaf_masks(const VoxelEdit *edits, std::uint32_t count,
                                    LeafMask *leaf_masks,
                                    std::uint32_t *leaf_count,
                                    const VoxelToLeafWorkspace &workspace,
                                    cudaStream_t stream) {
    if (count == 0u)
        return cudaMemsetAsync(leaf_count, 0, sizeof(std::uint32_t), stream);

    const std::uint32_t blocks = block_count(count, kKernelBlockSize);
    pack_voxel_edit_records_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
        edits, count, workspace.keys, workspace.local_bits);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    // Sorting groups all voxels in the same leaf so the selected implementation
    // can emit one LeafMask per unique leaf key.
    status = algo::cuda::sort::sort_pairs(
        workspace.keys, workspace.local_bits, count, workspace.sort_workspace,
        workspace.sort_workspace_size, stream);
    if (status != cudaSuccess)
        return status;

    return build_leaf_masks_impl(workspace, leaf_masks, leaf_count, count,
                                 stream);
}

} // namespace algo::svt::cuda::detail
