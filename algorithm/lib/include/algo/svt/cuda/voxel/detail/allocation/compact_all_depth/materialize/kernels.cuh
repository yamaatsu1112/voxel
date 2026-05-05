#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/traits.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

__device__ inline std::uint32_t
upper_bound_offset_rank(const std::uint32_t* offsets, std::uint32_t count,
                        std::uint32_t rank) {
    std::uint32_t first = 0u;
    std::uint32_t remaining = count;
    while (remaining != 0u) {
        const std::uint32_t step = remaining >> 1u;
        const std::uint32_t mid = first + step;
        if (offsets[mid] <= rank) {
            first = mid + 1u;
            remaining -= step + 1u;
        } else {
            remaining = step;
        }
    }
    return first;
}

template <class StartDepthMode, class Workspace>
__device__ inline std::uint32_t
node_parent_index(DeviceGpuSvo svo, const LeafMask* leaf_masks,
                  std::uint32_t leaf_count, const Workspace& workspace,
                  std::uint32_t owner, std::uint32_t depth,
                  std::uint32_t start_depth, std::uint32_t node_base) {
    using Traits = RankTraits<StartDepthMode>;
    const std::uint32_t first_missing_depth =
        missing_state_first_missing_depth(workspace.missing_states[owner]);
    if (depth > start_depth) {
        return materialized_node_index(
            svo, workspace.allocation_state,
            node_base + (depth - start_depth - 1u));
    }
    if (start_depth == first_missing_depth)
        return workspace.existing_parent_indices[owner];

    const std::uint32_t leaf_key = leaf_masks[owner].leaf_key;
    const std::uint32_t parent_depth = depth - 1u;
    const std::uint32_t parent_prefix =
        prefix_for_leaf_key(leaf_key, parent_depth);
    const std::uint32_t shift = (kMaxDepth - parent_depth) * kGroupSizeExp;
    const std::uint32_t first_key = parent_prefix << shift;
    const std::uint32_t parent_owner =
        lower_bound_leaf_key(leaf_masks, leaf_count, first_key);
    if (parent_owner >= leaf_count)
        return svo.max_node_count;

    return materialized_node_index(
        svo, workspace.allocation_state,
        Traits::emitted_node_rank(workspace, parent_owner, parent_depth));
}

template <class StartDepthMode, class Workspace>
__device__ inline void link_one_node(DeviceGpuSvo svo,
                                     const LeafMask* leaf_masks,
                                     std::uint32_t leaf_count,
                                     const Workspace& workspace,
                                     std::uint32_t owner, std::uint32_t depth,
                                     std::uint32_t node_rank) {
    using Traits = RankTraits<StartDepthMode>;
    const std::uint32_t start_depth = Traits::start_depth(workspace, owner);
    const std::uint32_t node_base =
        Traits::node_base(workspace, owner, start_depth);
    const std::uint32_t child_storage_index =
        materialized_node_index(svo, workspace.allocation_state, node_rank);
    if (child_storage_index >= svo.max_node_count)
        return;

    const std::uint32_t parent_index =
        node_parent_index<StartDepthMode>(svo, leaf_masks, leaf_count,
                                          workspace, owner, depth, start_depth,
                                          node_base);
    if (parent_index >= svo.max_node_count)
        return;

    const bool filled =
        missing_state_inherited_filled(workspace.missing_states[owner]) != 0u;
    const std::uint32_t child_index =
        child_index_for_leaf_key(leaf_masks[owner].leaf_key, depth - 1u);
    svo.nodes[parent_index].child_data[child_index] =
        make_child_data(child_storage_index, filled);
}

template <class StartDepthMode, class Workspace>
__device__ inline void link_one_leaf(DeviceGpuSvo svo,
                                     const LeafMask* leaf_masks,
                                     std::uint32_t leaf_count,
                                     const Workspace& workspace,
                                     std::uint32_t owner,
                                     std::uint32_t leaf_rank) {
    using Traits = RankTraits<StartDepthMode>;
    const std::uint32_t start_depth = Traits::start_depth(workspace, owner);
    const std::uint32_t node_base =
        Traits::node_base(workspace, owner, start_depth);
    const std::uint32_t parent_index =
        node_parent_index<StartDepthMode>(svo, leaf_masks, leaf_count,
                                          workspace, owner, kMaxDepth,
                                          start_depth, node_base);
    if (parent_index >= svo.max_node_count)
        return;

    const std::uint32_t child_storage_index =
        materialized_leaf_index(svo, workspace.allocation_state, leaf_rank);
    if (child_storage_index >= svo.max_leaf_count)
        return;

    const bool filled =
        missing_state_inherited_filled(workspace.missing_states[owner]) != 0u;
    const std::uint32_t child_index =
        child_index_for_leaf_key(leaf_masks[owner].leaf_key, kMaxDepth - 1u);
    svo.nodes[parent_index].child_data[child_index] =
        make_child_data(child_storage_index, filled);
}

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
