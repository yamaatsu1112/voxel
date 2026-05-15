#pragma once

#include <algo/svt/cuda/detail/edit_common.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

// Set payload bits for leaves that are already materialized by the allocation
// stage. If a path disappeared because another stage collapsed it, the mask is
// skipped instead of recreating topology here.
__global__ void update_leaf_values_kernel(DeviceGpuSvo svo,
                                          const LeafMask* leaf_masks,
                                          const std::uint32_t* leaf_count,
                                          std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity || index >= *leaf_count)
        return;

    const LeafMask mask = leaf_masks[index];
    std::uint32_t node_index = kRootNodeIndex;
    for (std::uint32_t depth = 0; depth < kMaxDepth; ++depth) {
        const std::uint32_t child_index =
            child_index_for_leaf_key(mask.leaf_key, depth);
        const GpuSvoNode node = svo.nodes[node_index];
        if (!node_has_child(node, child_index))
            return;
        node_index = node_child_index(node, child_index);
    }

    svo.leaves[node_index].voxel_data_low |= mask.voxel_data_low;
    svo.leaves[node_index].voxel_data_high |= mask.voxel_data_high;
}

// Clear payload bits for materialized leaves. Topology cleanup is deferred to
// the collapse stage so this kernel only mutates leaf masks.
__global__ void clear_leaf_values_kernel(DeviceGpuSvo svo,
                                         const LeafMask* leaf_masks,
                                         const std::uint32_t* leaf_count,
                                         std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity || index >= *leaf_count)
        return;

    const LeafMask mask = leaf_masks[index];
    std::uint32_t node_index = kRootNodeIndex;
    for (std::uint32_t depth = 0; depth < kMaxDepth; ++depth) {
        const std::uint32_t child_index =
            child_index_for_leaf_key(mask.leaf_key, depth);
        const GpuSvoNode node = svo.nodes[node_index];
        if (!node_has_child(node, child_index))
            return;
        node_index = node_child_index(node, child_index);
    }

    svo.leaves[node_index].voxel_data_low &= ~mask.voxel_data_low;
    svo.leaves[node_index].voxel_data_high &= ~mask.voxel_data_high;
}

inline cudaError_t apply_leaf_masks(DeviceGpuSvo svo, const LeafMask* leaf_masks,
                                    const std::uint32_t* leaf_count,
                                    std::uint32_t leaf_mask_capacity,
                                    EditOp op, cudaStream_t stream) {
    if (leaf_mask_capacity == 0u)
        return cudaSuccess;

    const std::uint32_t blocks =
        block_count(leaf_mask_capacity, kKernelBlockSize);
    switch (op) {
    case EditOp::Place:
        update_leaf_values_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
            svo, leaf_masks, leaf_count, leaf_mask_capacity);
        break;
    case EditOp::Destroy:
        clear_leaf_values_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
            svo, leaf_masks, leaf_count, leaf_mask_capacity);
        break;
    }
    return last_launch_status();
}

} // namespace algo::svt::cuda::detail
