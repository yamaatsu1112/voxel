#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/materialize_common.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

// A non-zero start depth means this leaf emits one materialized path suffix:
// internal nodes for [start_depth, kMaxDepth) and one leaf at kMaxDepth.
__device__ inline std::uint32_t node_emit_count(std::uint32_t start_depth) {
    return start_depth != 0u && start_depth < kMaxDepth
               ? kMaxDepth - start_depth
               : 0u;
}

__device__ inline std::uint32_t leaf_emit_count(std::uint32_t start_depth) {
    return start_depth != 0u ? 1u : 0u;
}

__device__ inline std::uint32_t lcp_depth(std::uint32_t lhs,
                                          std::uint32_t rhs) {
    std::uint32_t depth = 0u;
    for (; depth < kMaxDepth; ++depth) {
        if (child_index_for_leaf_key(lhs, depth) !=
            child_index_for_leaf_key(rhs, depth)) {
            break;
        }
    }
    return depth;
}

__device__ inline std::uint32_t lower_bound_leaf_key(const LeafMask* leaf_masks,
                                                     std::uint32_t count,
                                                     std::uint32_t key) {
    std::uint32_t first = 0u;
    std::uint32_t remaining = count;
    while (remaining != 0u) {
        const std::uint32_t step = remaining >> 1u;
        const std::uint32_t mid = first + step;
        if (leaf_masks[mid].leaf_key < key) {
            first = mid + 1u;
            remaining -= step + 1u;
        } else {
            remaining = step;
        }
    }
    return first;
}

__global__ void commit_materialized_allocation_state_kernel(
    DeviceGpuSvo svo, const std::uint32_t* node_offsets,
    const std::uint32_t* leaf_offsets, std::uint32_t leaf_mask_capacity,
    const AllocationState* state) {
    if (blockIdx.x != 0u || threadIdx.x != 0u)
        return;

    const std::uint32_t node_count =
        leaf_mask_capacity == 0u ? 0u : node_offsets[leaf_mask_capacity - 1u];
    const std::uint32_t leaf_count =
        leaf_mask_capacity == 0u ? 0u : leaf_offsets[leaf_mask_capacity - 1u];

    const std::uint32_t reused_nodes = node_count < state->base_free_node_count
                                           ? node_count
                                           : state->base_free_node_count;
    const std::uint32_t reused_leaves = leaf_count < state->base_free_leaf_count
                                            ? leaf_count
                                            : state->base_free_leaf_count;
    svo.counters->node_count =
        state->base_node_count + (node_count - reused_nodes);
    svo.counters->free_node_count = state->base_free_node_count - reused_nodes;
    svo.counters->leaf_count =
        state->base_leaf_count + (leaf_count - reused_leaves);
    svo.counters->free_leaf_count = state->base_free_leaf_count - reused_leaves;
}

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
