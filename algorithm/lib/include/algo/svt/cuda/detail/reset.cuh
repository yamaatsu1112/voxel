#pragma once

#include <algo/svt/cuda/device_view.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__global__ void reset_svo_kernel(DeviceGpuSvo svo) {
    if (blockIdx.x != 0u || threadIdx.x != 0u)
        return;

    for (std::uint32_t i = 0; i < kGroupSize; ++i)
        svo.nodes[kRootNodeIndex].child_data[i] = 0u;

    svo.counters->node_count = 1u;
    svo.counters->leaf_count = 0u;
    svo.counters->free_node_count = 0u;
    svo.counters->free_leaf_count = 0u;
}

} // namespace algo::svt::cuda::detail
