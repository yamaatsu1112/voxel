#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/traits.cuh>
#include <algo/svt/cuda/detail/launch.cuh>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

template <class InitMode, class StartDepthMode> struct Initializer;
template <class InitMode> struct InitLaunchTraits;

template <> struct InitLaunchTraits<Threadwise> {
    static constexpr std::uint32_t kItemsPerBlock = kKernelBlockSize;

    static dim3 block() { return dim3(kItemsPerBlock); }

    static dim3 grid(std::uint32_t item_count) {
        return dim3(block_count(item_count, kItemsPerBlock));
    }

    __device__ static std::uint32_t item_index() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ static bool should_initialize_leaf() { return true; }
};

template <> struct InitLaunchTraits<Childwise> {
    static constexpr std::uint32_t kItemsPerBlock = 32u;

    static dim3 block() { return dim3(kGroupSize, kItemsPerBlock); }

    static dim3 grid(std::uint32_t item_count) {
        return dim3(block_count(item_count, kItemsPerBlock));
    }

    __device__ static std::uint32_t item_index() {
        return blockIdx.x * blockDim.y + threadIdx.y;
    }

    __device__ static std::uint32_t child_index() { return threadIdx.x; }

    __device__ static bool should_initialize_leaf() {
        return child_index() == 0u;
    }
};

template <class StartDepthMode> struct Initializer<Threadwise, StartDepthMode> {
    template <class Workspace>
    __device__ static void initialize_node(DeviceGpuSvo svo,
                                           const Workspace& workspace,
                                           std::uint32_t owner,
                                           std::uint32_t node_rank) {
        const std::uint32_t node_index =
            materialized_node_index(svo, workspace.allocation_state, node_rank);
        if (node_index >= svo.max_node_count)
            return;

        const bool filled =
            missing_state_inherited_filled(workspace.missing_states[owner]) !=
            0u;
        GpuSvoNode node{};
        for (std::uint32_t child = 0u; child < kGroupSize; ++child)
            node.child_data[child] = make_uniform_child_data(filled);
        svo.nodes[node_index] = node;
    }

    template <class Workspace>
    __device__ static void initialize_leaf(DeviceGpuSvo svo,
                                           const Workspace& workspace,
                                           std::uint32_t owner,
                                           std::uint32_t leaf_rank) {
        const std::uint32_t leaf_index =
            materialized_leaf_index(svo, workspace.allocation_state, leaf_rank);
        if (leaf_index >= svo.max_leaf_count)
            return;

        const bool filled =
            missing_state_inherited_filled(workspace.missing_states[owner]) !=
            0u;
        svo.leaves[leaf_index] = filled ? GpuSvoLeaf{0xffffffffu, 0xffffffffu}
                                        : GpuSvoLeaf{0u, 0u};
    }
};

template <class StartDepthMode> struct Initializer<Childwise, StartDepthMode> {
    template <class Workspace>
    __device__ static void initialize_node(DeviceGpuSvo svo,
                                           const Workspace& workspace,
                                           std::uint32_t owner,
                                           std::uint32_t node_rank) {
        const std::uint32_t node_index =
            materialized_node_index(svo, workspace.allocation_state, node_rank);
        if (node_index >= svo.max_node_count)
            return;

        const bool filled =
            missing_state_inherited_filled(workspace.missing_states[owner]) !=
            0u;
        svo.nodes[node_index]
            .child_data[InitLaunchTraits<Childwise>::child_index()] =
            make_uniform_child_data(filled);
    }

    template <class Workspace>
    __device__ static void initialize_leaf(DeviceGpuSvo svo,
                                           const Workspace& workspace,
                                           std::uint32_t owner,
                                           std::uint32_t leaf_rank) {
        Initializer<Threadwise, StartDepthMode>::initialize_leaf(
            svo, workspace, owner, leaf_rank);
    }
};

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
