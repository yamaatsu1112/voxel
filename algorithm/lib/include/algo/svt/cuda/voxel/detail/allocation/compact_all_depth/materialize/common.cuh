#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

inline cudaError_t resolve_materialized_node_request_count(
    const std::uint32_t* node_offsets, std::uint32_t leaf_mask_capacity,
    std::uint32_t* node_request_count, cudaStream_t stream) {
    if (leaf_mask_capacity == 0u) {
        *node_request_count = 0u;
        return cudaSuccess;
    }

    cudaError_t status =
        cudaMemcpyAsync(node_request_count,
                        node_offsets + leaf_mask_capacity - 1u,
                        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess)
        return status;
    return cudaStreamSynchronize(stream);
}

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
