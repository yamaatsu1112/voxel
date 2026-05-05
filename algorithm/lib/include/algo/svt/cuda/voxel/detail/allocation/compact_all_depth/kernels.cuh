#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/initialize.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/traits.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

template <class StartDepthMode, class Workspace>
__global__ void
collect_materialize_counts_kernel(DeviceGpuSvo svo, const LeafMask* leaf_masks,
                                  const std::uint32_t* leaf_count_ptr,
                                  std::uint32_t leaf_mask_capacity,
                                  Workspace workspace) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity)
        return;

    const std::uint32_t leaf_count = *leaf_count_ptr;
    std::uint32_t first_missing_depth = 0u;
    std::uint32_t existing_parent_index = kRootNodeIndex;
    std::uint32_t filled = 0u;
    std::uint32_t start_depth = 0u;

    if (index < leaf_count) {
        const std::uint32_t leaf_key = leaf_masks[index].leaf_key;
        std::uint32_t node_index = kRootNodeIndex;
        for (std::uint32_t depth = 1u; depth <= kMaxDepth; ++depth) {
            const std::uint32_t child_index =
                child_index_for_leaf_key(leaf_key, depth - 1u);
            const GpuSvoNode node = svo.nodes[node_index];
            if (!node_has_child(node, child_index)) {
                first_missing_depth = depth;
                existing_parent_index = node_index;
                filled = node_is_filled(node, child_index) ? 1u : 0u;
                break;
            }
            node_index = node_child_index(node, child_index);
        }

        if (first_missing_depth != 0u) {
            const std::uint32_t lcp =
                index == 0u
                    ? 0u
                    : lcp_depth(leaf_key, leaf_masks[index - 1u].leaf_key);
            const std::uint32_t unique_depth = lcp + 1u;
            start_depth = first_missing_depth > unique_depth
                              ? first_missing_depth
                              : unique_depth;
        }
    }

    RankTraits<StartDepthMode>::record_start_depth(workspace, index,
                                                   start_depth);
    workspace.missing_states[index] =
        pack_missing_state(first_missing_depth, filled);
    workspace.existing_parent_indices[index] = existing_parent_index;
    workspace.node_offsets[index] = node_emit_count(start_depth);
    workspace.leaf_offsets[index] = leaf_emit_count(start_depth);
}

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
