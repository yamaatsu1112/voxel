#pragma once

#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>

#include <cstdint>

namespace algo::svt::cuda {

__device__ inline bool get_voxel(DeviceGpuSvo svo, std::uint32_t x,
                                 std::uint32_t y, std::uint32_t z) {
    if (!detail::valid_world_coord(x) || !detail::valid_world_coord(y) ||
        !detail::valid_world_coord(z)) {
        return false;
    }

    // Walk internal nodes using the high coordinate bits. A missing child means
    // the whole child region is represented by the node's filled bit.
    std::uint32_t node_index = kRootNodeIndex;
    for (std::uint32_t depth = 0; depth < kMaxDepth; ++depth) {
        const GpuSvoNode node = svo.nodes[node_index];
        const std::uint32_t child_index =
            detail::child_index_for_voxel(x, y, z, depth);
        if (!detail::node_has_child(node, child_index))
            return detail::node_is_filled(node, child_index);
        node_index = detail::node_child_index(node, child_index);
    }

    // After kMaxDepth steps node_index refers to the leaf payload.
    const std::uint32_t offset = detail::leaf_offset_for_voxel(x, y, z);
    return detail::has_mask_bit(svo.leaves[node_index].voxel_data_low,
                                svo.leaves[node_index].voxel_data_high,
                                offset);
}

} // namespace algo::svt::cuda
