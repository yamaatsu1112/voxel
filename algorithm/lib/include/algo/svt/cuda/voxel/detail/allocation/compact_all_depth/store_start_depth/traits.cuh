#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/traits.cuh>
#include <algo/svt/cuda/voxel/edit_config.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

namespace store_start_depth {

__device__ inline std::uint32_t
base_node_rank(const std::uint32_t* node_offsets, std::uint32_t index,
               std::uint32_t start_depth) {
    return node_offsets[index] - node_emit_count(start_depth);
}

__device__ inline std::uint32_t
base_leaf_rank(const std::uint32_t* leaf_offsets, std::uint32_t index,
               std::uint32_t start_depth) {
    return leaf_offsets[index] - leaf_emit_count(start_depth);
}

} // namespace store_start_depth

template <> struct RankTraits<StoreStartDepth> {
    template <class Workspace>
    __device__ static std::uint32_t start_depth(const Workspace& workspace,
                                                std::uint32_t index) {
        return workspace.start_depths[index];
    }

    template <class Workspace>
    __device__ static std::uint32_t node_base(const Workspace& workspace,
                                              std::uint32_t index,
                                              std::uint32_t start) {
        return store_start_depth::base_node_rank(workspace.node_offsets, index,
                                                 start);
    }

    template <class Workspace>
    __device__ static std::uint32_t leaf_base(const Workspace& workspace,
                                              std::uint32_t index,
                                              std::uint32_t start) {
        return store_start_depth::base_leaf_rank(workspace.leaf_offsets, index,
                                                 start);
    }

    template <class Workspace>
    __device__ static std::uint32_t
    emitted_node_rank(const Workspace& workspace, std::uint32_t leaf_index,
                      std::uint32_t depth) {
        const std::uint32_t start = start_depth(workspace, leaf_index);
        return node_base(workspace, leaf_index, start) + (depth - start);
    }

    template <class Workspace>
    __device__ static void record_start_depth(const Workspace& workspace,
                                              std::uint32_t index,
                                              std::uint32_t start) {
        workspace.start_depths[index] = start;
    }
};

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
