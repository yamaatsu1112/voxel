#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/all_depth/workspace.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::all_depth {

__global__ void collect_materialize_flags_kernel(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count_ptr, std::uint32_t leaf_mask_capacity,
    std::uint32_t* missing_states,
    std::uint32_t* existing_parent_indices,
    std::uint32_t* materialize_offsets) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity)
        return;

    const std::uint32_t leaf_count = *leaf_count_ptr;
    std::uint32_t first_missing_depth = 0u;
    std::uint32_t existing_parent_index = kRootNodeIndex;
    std::uint32_t filled = 0u;
    const bool valid_leaf = index < leaf_count;
    const std::uint32_t leaf_key =
        valid_leaf ? leaf_masks[index].leaf_key : 0u;
    const bool has_previous_leaf = index != 0u && valid_leaf;
    const std::uint32_t previous_leaf_key =
        has_previous_leaf ? leaf_masks[index - 1u].leaf_key : 0u;

    if (valid_leaf) {
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
    }

    missing_states[index] = pack_missing_state(first_missing_depth, filled);
    existing_parent_indices[index] = existing_parent_index;

    for (std::uint32_t depth = 1u; depth <= kMaxDepth; ++depth) {
        const std::uint32_t slot =
            (depth - 1u) * leaf_mask_capacity + index;
        const bool emits =
            valid_leaf &&
            materialize_slot_emits(leaf_key, previous_leaf_key,
                                   has_previous_leaf, first_missing_depth,
                                   depth);
        materialize_offsets[slot] = emits ? 1u : 0u;
    }
}

__global__ void compact_materialize_requests_kernel(
    const LeafMask* leaf_masks, const std::uint32_t* leaf_count_ptr,
    std::uint32_t leaf_mask_capacity,
    const std::uint32_t* missing_states,
    const std::uint32_t* existing_parent_indices,
    const std::uint32_t* materialize_offsets,
    MaterializeRequest* materialize_requests, std::uint32_t* materialize_count,
    std::uint32_t* materialize_node_count) {
    const std::uint32_t candidate_count =
        leaf_mask_capacity * kMaxDepth;
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t leaf_count = *leaf_count_ptr;

    if (index == 0u) {
        const std::uint32_t last = candidate_count - 1u;
        const std::uint32_t first_leaf_slot =
            (kMaxDepth - 1u) * leaf_mask_capacity;
        *materialize_count = materialize_offsets[last];
        *materialize_node_count =
            first_leaf_slot == 0u ? 0u
                                  : materialize_offsets[first_leaf_slot - 1u];
    }

    if (index >= candidate_count)
        return;

    const std::uint32_t depth = index / leaf_mask_capacity + 1u;
    const std::uint32_t leaf_index =
        index - ((depth - 1u) * leaf_mask_capacity);
    if (leaf_index >= leaf_count)
        return;

    const std::uint32_t missing_state = missing_states[leaf_index];
    const std::uint32_t first_missing_depth =
        missing_state_first_missing_depth(missing_state);

    const std::uint32_t leaf_key = leaf_masks[leaf_index].leaf_key;
    const bool has_previous_leaf = leaf_index != 0u;
    const std::uint32_t previous_leaf_key =
        has_previous_leaf ? leaf_masks[leaf_index - 1u].leaf_key : 0u;
    if (!materialize_slot_emits(leaf_key, previous_leaf_key,
                                has_previous_leaf, first_missing_depth,
                                depth)) {
        return;
    }

    std::uint32_t parent_ref = existing_parent_indices[leaf_index];
    bool parent_ref_is_request = false;
    if (depth > first_missing_depth) {
        const std::uint32_t parent_depth = depth - 1u;
        const std::uint32_t parent_candidate_slot =
            (parent_depth - 1u) * leaf_mask_capacity + leaf_index;
        parent_ref = materialize_offsets[parent_candidate_slot] - 1u;
        parent_ref_is_request = true;
    }

    const std::uint32_t output = materialize_offsets[index] - 1u;
    materialize_requests[output] =
        MaterializeRequest{
            pack_materialize_request_depth(
                depth, missing_state_inherited_filled(missing_state)),
            prefix_for_leaf_key(leaf_key, depth),
            parent_ref, materialize_request_metadata(parent_ref_is_request)};
}

__global__ void initialize_materialized_storage_kernel(
    DeviceGpuSvo svo, const MaterializeRequest* requests,
    const std::uint32_t* request_count,
    const std::uint32_t* node_request_count, const AllocationState* state) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *request_count;
    if (index >= count)
        return;

        const MaterializeRequest request = requests[index];
        const std::uint32_t depth = materialize_request_depth(request);
        const bool filled = materialize_request_filled(request);
        if (depth < kMaxDepth) {
        const std::uint32_t new_node_index =
            materialized_node_index(svo, state, index);
        if (new_node_index >= svo.max_node_count)
            return;

        GpuSvoNode new_node{};
        for (std::uint32_t i = 0; i < kGroupSize; ++i)
            new_node.child_data[i] = make_uniform_child_data(filled);
        svo.nodes[new_node_index] = new_node;
        } else {
            const std::uint32_t leaf_rank = index - *node_request_count;
        const std::uint32_t new_leaf_index =
            materialized_leaf_index(svo, state, leaf_rank);
        if (new_leaf_index >= svo.max_leaf_count)
            return;

        svo.leaves[new_leaf_index] =
            filled ? GpuSvoLeaf{0xffffffffu, 0xffffffffu}
                   : GpuSvoLeaf{0u, 0u};
    }
}

__global__ void link_materialized_storage_kernel(
    DeviceGpuSvo svo, const MaterializeRequest* requests,
    const std::uint32_t* request_count,
    const std::uint32_t* node_request_count,
    const AllocationState* state) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *request_count;
    if (index >= count)
        return;

    const MaterializeRequest request = requests[index];
    const std::uint32_t depth = materialize_request_depth(request);
    const bool filled = materialize_request_filled(request);
    std::uint32_t child_storage_index = 0u;
    if (depth < kMaxDepth) {
        child_storage_index = materialized_node_index(svo, state, index);
        if (child_storage_index >= svo.max_node_count)
            return;
    } else {
        const std::uint32_t leaf_rank = index - *node_request_count;
        child_storage_index = materialized_leaf_index(svo, state, leaf_rank);
        if (child_storage_index >= svo.max_leaf_count)
            return;
    }

    std::uint32_t parent_index = request.parent_ref;
    const std::uint32_t child_index = request.prefix & (kGroupSize - 1u);
    if (materialize_request_parent_ref_is_request(request)) {
        parent_index = materialized_node_index(svo, state, request.parent_ref);
    }

    if (parent_index >= svo.max_node_count)
        return;

    svo.nodes[parent_index].child_data[child_index] =
        make_child_data(child_storage_index, filled);
}

__global__ void commit_materialized_allocation_state_kernel(
    DeviceGpuSvo svo, const std::uint32_t* node_request_count,
    const std::uint32_t* request_count, const AllocationState* state) {
    if (blockIdx.x != 0u || threadIdx.x != 0u)
        return;

    const std::uint32_t node_count = *node_request_count;
    const std::uint32_t leaf_count = *request_count - node_count;

    const std::uint32_t reused_nodes =
        node_count < state->base_free_node_count ? node_count
                                                 : state->base_free_node_count;
    const std::uint32_t reused_leaves =
        leaf_count < state->base_free_leaf_count ? leaf_count
                                                 : state->base_free_leaf_count;
    svo.counters->node_count =
        state->base_node_count + (node_count - reused_nodes);
    svo.counters->free_node_count = state->base_free_node_count - reused_nodes;
    svo.counters->leaf_count =
        state->base_leaf_count + (leaf_count - reused_leaves);
    svo.counters->free_leaf_count = state->base_free_leaf_count - reused_leaves;
}

} // namespace allocation::all_depth

} // namespace algo::svt::cuda::detail
