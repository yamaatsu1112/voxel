#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/materialize_common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/scan_depthwise/workspace.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__global__ void collect_missing_state_kernel(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count_ptr, std::uint32_t leaf_mask_capacity,
    std::uint32_t* missing_states,
    std::uint32_t* existing_parent_indices) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity)
        return;

    const std::uint32_t leaf_count = *leaf_count_ptr;
    std::uint32_t first_missing_depth = 0u;
    std::uint32_t existing_parent_index = kRootNodeIndex;
    std::uint32_t filled = 0u;
    const bool valid_leaf = index < leaf_count;

    if (valid_leaf) {
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
    }

    missing_states[index] = pack_missing_state(first_missing_depth, filled);
    existing_parent_indices[index] = existing_parent_index;
}

__global__ void mark_depthwise_request_emits_kernel(
    const LeafMask* leaf_masks, const std::uint32_t* leaf_count_ptr,
    std::uint32_t leaf_mask_capacity, const std::uint32_t* missing_states,
    std::uint32_t* request_emits, std::uint32_t depth) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity)
        return;

    const std::uint32_t leaf_count = *leaf_count_ptr;
    if (index >= leaf_count) {
        request_emits[index] = 0u;
        return;
    }

    const std::uint32_t leaf_key = leaf_masks[index].leaf_key;
    const bool has_previous_leaf = index != 0u;
    const std::uint32_t previous_leaf_key =
        has_previous_leaf ? leaf_masks[index - 1u].leaf_key : 0u;
    const std::uint32_t first_missing_depth =
        missing_state_first_missing_depth(missing_states[index]);
    request_emits[index] =
        materialize_slot_emits(leaf_key, previous_leaf_key, has_previous_leaf,
                               first_missing_depth, depth)
            ? 1u
            : 0u;
}

__global__ void compact_depthwise_requests_kernel(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count_ptr, std::uint32_t leaf_mask_capacity,
    const std::uint32_t* missing_states,
    const std::uint32_t* existing_parent_indices,
    const std::uint32_t* previous_request_emits,
    const std::uint32_t* current_request_emits,
    const AllocationState* previous_allocation_state,
    AllocationRequest* unique_requests, std::uint32_t* unique_count,
    std::uint32_t depth) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t leaf_count = *leaf_count_ptr;

    if (index == leaf_mask_capacity - 1u)
        *unique_count = current_request_emits[index];

    if (index >= leaf_mask_capacity || index >= leaf_count)
        return;

    const std::uint32_t leaf_key = leaf_masks[index].leaf_key;
    const bool has_previous_leaf = index != 0u;
    const std::uint32_t previous_leaf_key =
        has_previous_leaf ? leaf_masks[index - 1u].leaf_key : 0u;
    const std::uint32_t first_missing_depth =
        missing_state_first_missing_depth(missing_states[index]);
    if (!materialize_slot_emits(leaf_key, previous_leaf_key, has_previous_leaf,
                                first_missing_depth, depth)) {
        return;
    }

    std::uint32_t parent_index = existing_parent_indices[index];
    if (depth > first_missing_depth) {
        const std::uint32_t parent_rank = previous_request_emits[index];
        parent_index = materialized_node_index(svo, previous_allocation_state,
                                               parent_rank - 1u);
    }

    const std::uint32_t request_index = current_request_emits[index] - 1u;
    unique_requests[request_index] = AllocationRequest{
        parent_index, child_index_for_leaf_key(leaf_key, depth - 1u)};
}

__global__ void allocate_scan_depth_batch_kernel(
    DeviceGpuSvo svo, const AllocationRequest* requests,
    const std::uint32_t* request_count, const AllocationState* state,
    std::uint32_t depth) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *request_count;
    if (index >= count)
        return;

    const AllocationRequest request = requests[index];
    GpuSvoNode parent = svo.nodes[request.node_index];
    const bool parent_filled = node_is_filled(parent, request.child_index);

    if (depth < kMaxDepth) {
        std::uint32_t new_node_index = 0u;
        if (index < state->base_free_node_count) {
            new_node_index =
                svo.free_node_indices[state->base_free_node_count - 1u - index];
        } else {
            new_node_index = state->base_node_count +
                             (index - state->base_free_node_count);
        }

        if (new_node_index >= svo.max_node_count)
            return;

        GpuSvoNode new_node{};
        for (std::uint32_t i = 0; i < kGroupSize; ++i)
            new_node.child_data[i] = make_uniform_child_data(parent_filled);
        svo.nodes[new_node_index] = new_node;
        svo.nodes[request.node_index].child_data[request.child_index] =
            make_child_data(new_node_index, parent_filled);

        if (index == 0u) {
            const std::uint32_t reused =
                count < state->base_free_node_count ? count
                                                    : state->base_free_node_count;
            const std::uint32_t new_count = count - reused;
            svo.counters->node_count = state->base_node_count + new_count;
            svo.counters->free_node_count =
                state->base_free_node_count - reused;
        }
    } else {
        std::uint32_t new_leaf_index = 0u;
        if (index < state->base_free_leaf_count) {
            new_leaf_index =
                svo.free_leaf_indices[state->base_free_leaf_count - 1u - index];
        } else {
            new_leaf_index = state->base_leaf_count +
                             (index - state->base_free_leaf_count);
        }

        if (new_leaf_index >= svo.max_leaf_count)
            return;

        svo.leaves[new_leaf_index] =
            parent_filled ? GpuSvoLeaf{0xffffffffu, 0xffffffffu}
                          : GpuSvoLeaf{0u, 0u};
        svo.nodes[request.node_index].child_data[request.child_index] =
            make_child_data(new_leaf_index, parent_filled);

        if (index == 0u) {
            const std::uint32_t reused =
                count < state->base_free_leaf_count ? count
                                                    : state->base_free_leaf_count;
            const std::uint32_t new_count = count - reused;
            svo.counters->leaf_count = state->base_leaf_count + new_count;
            svo.counters->free_leaf_count =
                state->base_free_leaf_count - reused;
        }
    }
}

} // namespace algo::svt::cuda::detail
