#pragma once

#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__global__ void collect_free_requests_kernel(
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
    std::uint32_t parent_index = kRootNodeIndex;
    std::uint32_t parent_child_index = 0u;

    if (depth == kMaxDepth) {
        for (std::uint32_t d = 0; d < kMaxDepth; ++d) {
            const std::uint32_t child_index =
                child_index_for_leaf_key(leaf_key, d);
            const GpuSvoNode node = svo.nodes[node_index];
            if (!node_has_child(node, child_index)) {
                request_keys[index] = kInvalidRequestKey;
                return;
            }
            if (d == kMaxDepth - 1u) {
                parent_index = node_index;
                parent_child_index = child_index;
            }
            node_index = node_child_index(node, child_index);
        }

        const GpuSvoLeaf leaf = svo.leaves[node_index];
        const bool empty =
            leaf.voxel_data_low == 0u && leaf.voxel_data_high == 0u;
        const bool full = leaf.voxel_data_low == 0xffffffffu &&
                          leaf.voxel_data_high == 0xffffffffu;
        if (!empty && !full) {
            request_keys[index] = kInvalidRequestKey;
            return;
        }
    } else {
        for (std::uint32_t d = 0; d < depth; ++d) {
            const std::uint32_t child_index =
                child_index_for_leaf_key(leaf_key, d);
            const GpuSvoNode node = svo.nodes[node_index];
            if (!node_has_child(node, child_index)) {
                request_keys[index] = kInvalidRequestKey;
                return;
            }
            if (d == depth - 1u) {
                parent_index = node_index;
                parent_child_index = child_index;
            }
            node_index = node_child_index(node, child_index);
        }

        const GpuSvoNode node = svo.nodes[node_index];
        bool all_empty = true;
        bool all_full = true;
        for (std::uint32_t i = 0; i < kGroupSize; ++i) {
            if (node_has_child(node, i)) {
                all_empty = false;
                all_full = false;
                break;
            }
            if (node_is_filled(node, i)) {
                all_empty = false;
            } else {
                all_full = false;
            }
        }

        if (!all_empty && !all_full) {
            request_keys[index] = kInvalidRequestKey;
            return;
        }
    }

    request_keys[index] = request_key_for_child(parent_index, parent_child_index);
}

} // namespace algo::svt::cuda::detail
