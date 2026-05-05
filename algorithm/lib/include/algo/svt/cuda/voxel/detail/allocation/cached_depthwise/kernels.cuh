#pragma once

#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__global__ void initialize_cached_parent_indices_kernel(
    std::uint32_t* parent_indices, const std::uint32_t* leaf_count,
    std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity)
        return;

    parent_indices[index] =
        index < *leaf_count ? kRootNodeIndex : kInvalidRequestKey;
}

__global__ void collect_cached_allocation_requests_kernel(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count, const std::uint32_t* parent_indices,
    std::uint32_t depth, std::uint32_t* request_keys,
    std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity)
        return;

    if (index >= *leaf_count || parent_indices[index] == kInvalidRequestKey) {
        request_keys[index] = kInvalidRequestKey;
        return;
    }

    const std::uint32_t leaf_key = leaf_masks[index].leaf_key;
    const std::uint32_t parent_index = parent_indices[index];
    const std::uint32_t child_index =
        child_index_for_leaf_key(leaf_key, depth - 1u);
    const GpuSvoNode node = svo.nodes[parent_index];
    if (node_has_child(node, child_index)) {
        request_keys[index] = kInvalidRequestKey;
        return;
    }

    request_keys[index] = request_key_for_child(parent_index, child_index);
}

__global__ void advance_cached_parent_indices_kernel(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count, std::uint32_t* parent_indices,
    std::uint32_t depth, std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity || index >= *leaf_count ||
        parent_indices[index] == kInvalidRequestKey) {
        return;
    }

    const std::uint32_t leaf_key = leaf_masks[index].leaf_key;
    const std::uint32_t parent_index = parent_indices[index];
    const std::uint32_t child_index =
        child_index_for_leaf_key(leaf_key, depth - 1u);
    const GpuSvoNode node = svo.nodes[parent_index];
    if (!node_has_child(node, child_index)) {
        parent_indices[index] = kInvalidRequestKey;
        return;
    }

    parent_indices[index] = node_child_index(node, child_index);
}

} // namespace algo::svt::cuda::detail
