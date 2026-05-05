#pragma once

#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__global__ void collect_allocation_requests_kernel(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count, std::uint32_t depth,
    std::uint32_t* request_keys, std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity)
        return;

    if (index >= *leaf_count) {
        request_keys[index] = kInvalidRequestKey;
        return;
    }

    const std::uint32_t leaf_key = leaf_masks[index].leaf_key;
    std::uint32_t node_index = kRootNodeIndex;
    for (std::uint32_t d = 0; d + 1u < depth; ++d) {
        const std::uint32_t child_index = child_index_for_leaf_key(leaf_key, d);
        const GpuSvoNode node = svo.nodes[node_index];
        if (!node_has_child(node, child_index)) {
            request_keys[index] = kInvalidRequestKey;
            return;
        }
        node_index = node_child_index(node, child_index);
    }

    const std::uint32_t child_index =
        child_index_for_leaf_key(leaf_key, depth - 1u);
    const GpuSvoNode node = svo.nodes[node_index];
    if (node_has_child(node, child_index)) {
        request_keys[index] = kInvalidRequestKey;
        return;
    }

    request_keys[index] = request_key_for_child(node_index, child_index);
}

} // namespace algo::svt::cuda::detail
