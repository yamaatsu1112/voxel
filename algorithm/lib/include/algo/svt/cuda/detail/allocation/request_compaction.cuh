#pragma once

#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/device_view.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__global__ void mark_unique_request_offsets_kernel(
    const std::uint32_t* request_keys, std::uint32_t* unique_offsets,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const bool valid = request_keys[index] != kInvalidRequestKey;
    const bool unique =
        valid && (index == 0u || request_keys[index - 1u] != request_keys[index]);
    unique_offsets[index] = unique ? 1u : 0u;
}

__global__ void compact_unique_requests_kernel(
    const std::uint32_t* request_keys, const std::uint32_t* unique_offsets,
    std::uint32_t source_count, AllocationRequest* unique_requests,
    std::uint32_t* unique_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= source_count)
        return;

    const bool valid = request_keys[index] != kInvalidRequestKey;
    const bool unique =
        valid && (index == 0u || request_keys[index - 1u] != request_keys[index]);
    if (unique) {
        const std::uint32_t request_key = request_keys[index];
        unique_requests[unique_offsets[index]] =
            AllocationRequest{request_key_node_index(request_key),
                              request_key_child_index(request_key)};
    }

    if (index == source_count - 1u)
        *unique_count = unique_offsets[index] + (unique ? 1u : 0u);
}

__global__ void snapshot_allocation_state_kernel(DeviceGpuSvo svo,
                                                 AllocationState* state) {
    if (blockIdx.x != 0u || threadIdx.x != 0u)
        return;

    state->base_node_count = svo.counters->node_count;
    state->base_leaf_count = svo.counters->leaf_count;
    state->base_free_node_count = svo.counters->free_node_count;
    state->base_free_leaf_count = svo.counters->free_leaf_count;
}

} // namespace algo::svt::cuda::detail
