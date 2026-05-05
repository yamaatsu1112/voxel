#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/initialize.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/materialize/common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/materialize/kernels.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/materialize/materializer.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/voxel/edit_config.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

template <class StartDepthMode, class Workspace>
__global__ void fill_explicit_requests_kernel(Workspace workspace,
                                              std::uint32_t leaf_mask_capacity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
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
        workspace.node_request_leaf_indices[rank] = index;
        workspace.node_request_depths[rank] = depth;
    }

    const std::uint32_t leaf_rank =
        Traits::leaf_base(workspace, index, start_depth);
    workspace.leaf_request_leaf_indices[leaf_rank] = index;
}

template <class InitMode, class StartDepthMode, class Workspace>
__global__ void initialize_nodes_explicit_kernel(
    DeviceGpuSvo svo, Workspace workspace, std::uint32_t leaf_mask_capacity,
    std::uint32_t node_request_count) {
    const std::uint32_t node_rank = InitLaunchTraits<InitMode>::item_index();
    if (node_rank >= node_request_count)
        return;

    const std::uint32_t total_nodes =
        workspace.node_offsets[leaf_mask_capacity - 1u];
    if (node_rank >= total_nodes)
        return;

    Initializer<InitMode, StartDepthMode>::initialize_node(
        svo, workspace, workspace.node_request_leaf_indices[node_rank],
        node_rank);
}

template <class StartDepthMode, class Workspace>
__global__ void link_nodes_explicit_kernel(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count_ptr, Workspace workspace,
    std::uint32_t leaf_mask_capacity, std::uint32_t node_request_count) {
    const std::uint32_t node_rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_rank >= node_request_count)
        return;

    const std::uint32_t total_nodes =
        workspace.node_offsets[leaf_mask_capacity - 1u];
    if (node_rank >= total_nodes)
        return;

    const std::uint32_t owner = workspace.node_request_leaf_indices[node_rank];
    const std::uint32_t depth = workspace.node_request_depths[node_rank];
    link_one_node<StartDepthMode>(svo, leaf_masks, *leaf_count_ptr, workspace,
                                  owner, depth, node_rank);
}

template <class InitMode, class StartDepthMode, class Workspace>
__global__ void initialize_leaves_explicit_kernel(
    DeviceGpuSvo svo, Workspace workspace, std::uint32_t leaf_mask_capacity) {
    const std::uint32_t leaf_rank = InitLaunchTraits<InitMode>::item_index();
    if (leaf_rank >= leaf_mask_capacity)
        return;
    if (!InitLaunchTraits<InitMode>::should_initialize_leaf())
        return;

    const std::uint32_t total_leaves =
        workspace.leaf_offsets[leaf_mask_capacity - 1u];
    if (leaf_rank >= total_leaves)
        return;

    Initializer<InitMode, StartDepthMode>::initialize_leaf(
        svo, workspace, workspace.leaf_request_leaf_indices[leaf_rank],
        leaf_rank);
}

template <class StartDepthMode, class Workspace>
__global__ void link_leaves_explicit_kernel(
    DeviceGpuSvo svo, const LeafMask* leaf_masks,
    const std::uint32_t* leaf_count_ptr, Workspace workspace,
    std::uint32_t leaf_mask_capacity) {
    const std::uint32_t leaf_rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_rank >= leaf_mask_capacity)
        return;

    const std::uint32_t total_leaves =
        workspace.leaf_offsets[leaf_mask_capacity - 1u];
    if (leaf_rank >= total_leaves)
        return;

    link_one_leaf<StartDepthMode>(
        svo, leaf_masks, *leaf_count_ptr, workspace,
        workspace.leaf_request_leaf_indices[leaf_rank], leaf_rank);
}

template <class InitMode, class StartDepthMode>
struct Materializer<Nodewise<ExplicitRequests>, InitMode, StartDepthMode> {
    template <class Workspace>
    static cudaError_t run(DeviceGpuSvo svo, const LeafMask* leaf_masks,
                           const std::uint32_t* leaf_count,
                           std::uint32_t leaf_mask_capacity,
                           Workspace& workspace, cudaStream_t stream) {
        const std::uint32_t leaf_blocks =
            block_count(leaf_mask_capacity, kKernelBlockSize);
        fill_explicit_requests_kernel<StartDepthMode>
            <<<leaf_blocks, kKernelBlockSize, 0, stream>>>(
                workspace, leaf_mask_capacity);
        cudaError_t status = last_launch_status();
        if (status != cudaSuccess)
            return status;

        std::uint32_t node_request_count = 0u;
        status = resolve_materialized_node_request_count(
            workspace.node_offsets, leaf_mask_capacity, &node_request_count,
            stream);
        if (status != cudaSuccess)
            return status;

        if (node_request_count != 0u) {
            initialize_nodes_explicit_kernel<InitMode, StartDepthMode>
                <<<InitLaunchTraits<InitMode>::grid(node_request_count),
                   InitLaunchTraits<InitMode>::block(), 0, stream>>>(
                    svo, workspace, leaf_mask_capacity, node_request_count);
            status = last_launch_status();
            if (status != cudaSuccess)
                return status;
        }

        initialize_leaves_explicit_kernel<InitMode, StartDepthMode>
            <<<InitLaunchTraits<InitMode>::grid(leaf_mask_capacity),
               InitLaunchTraits<InitMode>::block(), 0, stream>>>(
                svo, workspace, leaf_mask_capacity);
        status = last_launch_status();
        if (status != cudaSuccess)
            return status;

        const std::uint32_t node_blocks =
            block_count(node_request_count, kKernelBlockSize);
        if (node_request_count != 0u) {
            link_nodes_explicit_kernel<StartDepthMode>
                <<<node_blocks, kKernelBlockSize, 0, stream>>>(
                    svo, leaf_masks, leaf_count, workspace, leaf_mask_capacity,
                    node_request_count);
            status = last_launch_status();
            if (status != cudaSuccess)
                return status;
        }

        link_leaves_explicit_kernel<StartDepthMode>
            <<<leaf_blocks, kKernelBlockSize, 0, stream>>>(
                svo, leaf_masks, leaf_count, workspace, leaf_mask_capacity);
        return last_launch_status();
    }
};

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
