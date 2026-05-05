#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/initialize.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/materialize/materializer.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/traits.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/voxel/edit_config.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

template <class InitMode, class StartDepthMode, class Workspace>
__global__ void initialize_materialized_storage_kernel(
    DeviceGpuSvo svo, Workspace workspace, std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = InitLaunchTraits<InitMode>::item_index();
    if (index >= leaf_mask_capacity)
        return;

    using Traits = RankTraits<StartDepthMode>;
    const std::uint32_t start_depth = Traits::start_depth(workspace, index);
    if (start_depth == 0u)
        return;

    const std::uint32_t node_base =
        Traits::node_base(workspace, index, start_depth);
    for (std::uint32_t depth = start_depth; depth < kMaxDepth; ++depth) {
        const std::uint32_t rank = node_base + (depth - start_depth);
        Initializer<InitMode, StartDepthMode>::initialize_node(
            svo, workspace, index, rank);
    }

    if (!InitLaunchTraits<InitMode>::should_initialize_leaf())
        return;

    const std::uint32_t leaf_rank =
        Traits::leaf_base(workspace, index, start_depth);
    Initializer<InitMode, StartDepthMode>::initialize_leaf(
        svo, workspace, index, leaf_rank);
}

template <class StartDepthMode, class Workspace>
__global__ void
link_materialized_storage_kernel(DeviceGpuSvo svo, const LeafMask* leaf_masks,
                                 const std::uint32_t* leaf_count_ptr,
                                 Workspace workspace,
                                 std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_mask_capacity)
        return;

    using Traits = RankTraits<StartDepthMode>;
    const std::uint32_t start_depth = Traits::start_depth(workspace, index);
    if (start_depth == 0u)
        return;

    const std::uint32_t leaf_count = *leaf_count_ptr;
    if (index >= leaf_count)
        return;

    const std::uint32_t first_missing_depth =
        missing_state_first_missing_depth(workspace.missing_states[index]);
    const bool filled =
        missing_state_inherited_filled(workspace.missing_states[index]) != 0u;
    const std::uint32_t leaf_key = leaf_masks[index].leaf_key;
    const std::uint32_t node_base =
        Traits::node_base(workspace, index, start_depth);

    for (std::uint32_t depth = start_depth; depth <= kMaxDepth; ++depth) {
        std::uint32_t child_storage_index = 0u;
        if (depth < kMaxDepth) {
            child_storage_index =
                materialized_node_index(svo, workspace.allocation_state,
                                        node_base + (depth - start_depth));
            if (child_storage_index >= svo.max_node_count)
                continue;
        } else {
            const std::uint32_t leaf_rank =
                Traits::leaf_base(workspace, index, start_depth);
            child_storage_index = materialized_leaf_index(
                svo, workspace.allocation_state, leaf_rank);
            if (child_storage_index >= svo.max_leaf_count)
                continue;
        }

        std::uint32_t parent_index = workspace.existing_parent_indices[index];
        if (depth > start_depth) {
            parent_index =
                materialized_node_index(svo, workspace.allocation_state,
                                        node_base + (depth - start_depth - 1u));
        } else if (start_depth > first_missing_depth) {
            const std::uint32_t parent_depth = depth - 1u;
            const std::uint32_t parent_prefix =
                prefix_for_leaf_key(leaf_key, parent_depth);
            const std::uint32_t shift =
                (kMaxDepth - parent_depth) * kGroupSizeExp;
            const std::uint32_t first_key = parent_prefix << shift;
            const std::uint32_t owner =
                lower_bound_leaf_key(leaf_masks, leaf_count, first_key);
            if (owner >= leaf_count)
                continue;

            parent_index = materialized_node_index(
                svo, workspace.allocation_state,
                Traits::emitted_node_rank(workspace, owner, parent_depth));
        }

        if (parent_index >= svo.max_node_count)
            continue;

        const std::uint32_t child_index =
            child_index_for_leaf_key(leaf_key, depth - 1u);
        svo.nodes[parent_index].child_data[child_index] =
            make_child_data(child_storage_index, filled);
    }
}

template <class InitMode, class StartDepthMode>
struct Materializer<LeafwiseMaterialize, InitMode, StartDepthMode> {
    template <class Workspace>
    static cudaError_t run(DeviceGpuSvo svo, const LeafMask* leaf_masks,
                           const std::uint32_t* leaf_count,
                           std::uint32_t leaf_mask_capacity,
                           Workspace& workspace, cudaStream_t stream) {
        initialize_materialized_storage_kernel<InitMode, StartDepthMode>
            <<<InitLaunchTraits<InitMode>::grid(leaf_mask_capacity),
               InitLaunchTraits<InitMode>::block(), 0, stream>>>(
                svo, workspace, leaf_mask_capacity);
        cudaError_t status = last_launch_status();
        if (status != cudaSuccess)
            return status;

        const std::uint32_t blocks =
            block_count(leaf_mask_capacity, kKernelBlockSize);
        link_materialized_storage_kernel<StartDepthMode>
            <<<blocks, kKernelBlockSize, 0, stream>>>(
                svo, leaf_masks, leaf_count, workspace, leaf_mask_capacity);
        return last_launch_status();
    }
};

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
