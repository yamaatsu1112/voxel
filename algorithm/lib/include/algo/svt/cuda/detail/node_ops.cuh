#pragma once

#include <algo/svt/cuda/device_view.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__device__ inline bool node_has_child(const GpuSvoNode& node,
                                      std::uint32_t child_index) {
    return (node.child_data[child_index] & kChildMaskBit) != 0u;
}

__device__ inline bool node_is_filled(const GpuSvoNode& node,
                                      std::uint32_t child_index) {
    return (node.child_data[child_index] & kFilledBit) != 0u;
}

__device__ inline std::uint32_t node_child_index(const GpuSvoNode& node,
                                                 std::uint32_t child_index) {
    return node.child_data[child_index] & kChildIndexMask;
}

__device__ inline std::uint32_t make_child_data(std::uint32_t child_index,
                                                bool filled) {
    return (child_index & kChildIndexMask) | kChildMaskBit |
           (filled ? kFilledBit : 0u);
}

__device__ inline std::uint32_t make_uniform_child_data(bool filled) {
    return filled ? kFilledBit : 0u;
}

} // namespace algo::svt::cuda::detail
