#pragma once

#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/edit_common.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/terminal/detail/release.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__device__ inline void record_terminal_detached_child(
    TerminalDetachedChild* detached_children,
    std::uint32_t* detached_child_count, std::uint32_t child_data,
    std::uint32_t child_depth, std::uint32_t detached_child_capacity,
    cudaError_t* status) {
    const std::uint32_t index = atomicAdd(detached_child_count, 1u);
    if (index >= detached_child_capacity) {
        *status = cudaErrorInvalidValue;
        return;
    }
    detached_children[index] = TerminalDetachedChild{child_data, child_depth};
}

__device__ inline bool find_terminal_child_node(DeviceGpuSvo svo,
                                                std::uint32_t parent_index,
                                                std::uint32_t child,
                                                std::uint32_t& child_index) {
    const std::uint32_t child_data =
        svo.nodes[parent_index].child_data[child];
    if (terminal_child_is_allocating(child_data))
        return false;
    if ((child_data & kChildMaskBit) == 0u)
        return false;
    child_index = child_data & kChildIndexMask;
    return true;
}

__device__ inline bool find_terminal_leaf(DeviceGpuSvo svo,
                                          std::uint32_t leaf_prefix,
                                          std::uint32_t& leaf_index) {
    std::uint32_t node_index = kRootNodeIndex;
    for (std::uint32_t depth = 0u; depth < kMaxDepth; ++depth) {
        const std::uint32_t child = child_index_for_leaf_key(leaf_prefix, depth);
        if (depth + 1u == kMaxDepth) {
            const std::uint32_t child_data =
                svo.nodes[node_index].child_data[child];
            if (terminal_child_is_allocating(child_data))
                return false;
            if ((child_data & kChildMaskBit) == 0u)
                return false;
            leaf_index = child_data & kChildIndexMask;
            return true;
        }

        if (!find_terminal_child_node(svo, node_index, child, node_index))
            return false;
    }
    return false;
}

__device__ inline bool find_terminal_cell_parent(
    DeviceGpuSvo svo, const CellWriteRequest& request,
    std::uint32_t& parent_index, std::uint32_t& child_index) {
    parent_index = kRootNodeIndex;
    if (request.level == 0u) {
        child_index = 0u;
        return true;
    }

    for (std::uint32_t depth = 0u; depth + 1u < request.level; ++depth) {
        const std::uint32_t shift =
            (request.level - depth - 1u) * kGroupSizeExp;
        const std::uint32_t child =
            static_cast<std::uint32_t>((request.prefix >> shift) &
                                       (kGroupSize - 1u));
        if (!find_terminal_child_node(svo, parent_index, child, parent_index))
            return false;
    }

    child_index =
        static_cast<std::uint32_t>(request.prefix & (kGroupSize - 1u));
    return true;
}

__device__ inline bool apply_terminal_brick_mask_preallocated(
    DeviceGpuSvo svo, TerminalBrickMask mask, bool filled) {
    std::uint32_t leaf_index = 0u;
    if (!find_terminal_leaf(svo, mask.leafPrefix, leaf_index))
        return false;

    const std::uint32_t mask_low = static_cast<std::uint32_t>(mask.mask64);
    const std::uint32_t mask_high =
        static_cast<std::uint32_t>(mask.mask64 >> 32u);
    if (filled) {
        atomicOr(&svo.leaves[leaf_index].voxel_data_low, mask_low);
        atomicOr(&svo.leaves[leaf_index].voxel_data_high, mask_high);
    } else {
        atomicAnd(&svo.leaves[leaf_index].voxel_data_low, ~mask_low);
        atomicAnd(&svo.leaves[leaf_index].voxel_data_high, ~mask_high);
    }
    return true;
}

__device__ inline bool apply_terminal_cell_write_preallocated(
    DeviceGpuSvo svo, CellWriteRequest request, bool filled,
    TerminalDetachedChild* detached_children,
    std::uint32_t* detached_child_count,
    std::uint32_t detached_child_capacity,
    cudaError_t* status) {
    if (request.level == 0u)
        return false;

    std::uint32_t parent_index = 0u;
    std::uint32_t child = 0u;
    if (!find_terminal_cell_parent(svo, request, parent_index, child))
        return false;
    std::uint32_t& child_data = svo.nodes[parent_index].child_data[child];
    const std::uint32_t old_child_data = child_data;
    child_data = make_uniform_child_data(filled);
    record_terminal_detached_child(detached_children, detached_child_count,
                                   old_child_data, request.level,
                                   detached_child_capacity, status);
    return true;
}

__global__ void apply_terminal_brick_requests_kernel(
    DeviceGpuSvo svo, const TerminalRequest* requests,
    std::uint32_t request_count, bool filled, cudaError_t* status) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= request_count)
        return;

    if (*status != cudaSuccess)
        return;

    const TerminalRequest request = requests[index];
    if (!terminal_request_is_brick(request))
        return;

    if (!apply_terminal_brick_mask_preallocated(
            svo, terminal_request_brick(request), filled)) {
        *status = cudaErrorMemoryAllocation;
    }
}

__global__ void apply_terminal_cell_requests_kernel(
    DeviceGpuSvo svo, const TerminalRequest* requests,
    std::uint32_t request_count, bool filled,
    cudaError_t* status, TerminalDetachedChild* detached_children,
    std::uint32_t* detached_child_count,
    std::uint32_t detached_child_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= request_count)
        return;

    if (*status != cudaSuccess)
        return;

    const TerminalRequest request = requests[index];
    if (!terminal_request_is_cell(request))
        return;

    const CellWriteRequest cell = terminal_request_cell(request);
    if (!apply_terminal_cell_write_preallocated(
            svo, cell, filled, detached_children, detached_child_count,
            detached_child_capacity, status)) {
        *status = cudaErrorMemoryAllocation;
    }
}

} // namespace algo::svt::cuda::detail
