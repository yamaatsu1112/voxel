#pragma once

#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

__global__ void allocate_batch_kernel(DeviceGpuSvo svo,
                                      const AllocationRequest* requests,
                                      const std::uint32_t* request_count,
                                      const AllocationState* state,
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

__global__ void free_batch_kernel(DeviceGpuSvo svo,
                                  const AllocationRequest* requests,
                                  const std::uint32_t* request_count,
                                  const AllocationState* state,
                                  std::uint32_t depth) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *request_count;
    if (index >= count)
        return;

    const AllocationRequest request = requests[index];
    GpuSvoNode parent = svo.nodes[request.node_index];
    const std::uint32_t child_storage_index =
        node_child_index(parent, request.child_index);

    if (depth == kMaxDepth) {
        const std::uint32_t free_list_index =
            state->base_free_leaf_count + index;
        if (free_list_index >= svo.max_leaf_count)
            return;

        svo.free_leaf_indices[free_list_index] = child_storage_index;

        const GpuSvoLeaf leaf = svo.leaves[child_storage_index];
        const bool full = leaf.voxel_data_low == 0xffffffffu &&
                          leaf.voxel_data_high == 0xffffffffu;
        svo.nodes[request.node_index].child_data[request.child_index] =
            make_uniform_child_data(full);

        if (index == 0u)
            svo.counters->free_leaf_count = state->base_free_leaf_count + count;
    } else {
        const std::uint32_t free_list_index =
            state->base_free_node_count + index;
        if (free_list_index >= svo.max_node_count)
            return;

        svo.free_node_indices[free_list_index] = child_storage_index;

        const GpuSvoNode node = svo.nodes[child_storage_index];
        bool full = true;
        for (std::uint32_t i = 0; i < kGroupSize; ++i) {
            if (node_has_child(node, i) || !node_is_filled(node, i)) {
                full = false;
                break;
            }
        }
        svo.nodes[request.node_index].child_data[request.child_index] =
            make_uniform_child_data(full);

        if (index == 0u)
            svo.counters->free_node_count = state->base_free_node_count + count;
    }
}

} // namespace algo::svt::cuda::detail
