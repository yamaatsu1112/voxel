#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/traits.cuh>
#include <algo/svt/cuda/voxel/edit_config.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

namespace recover_start_depth {

__device__ inline std::uint32_t
scanned_node_emit_count(const std::uint32_t* node_offsets,
                        std::uint32_t index) {
    return node_offsets[index] - (index == 0u ? 0u : node_offsets[index - 1u]);
}

__device__ inline std::uint32_t
scanned_leaf_emit_count(const std::uint32_t* leaf_offsets,
                        std::uint32_t index) {
    return leaf_offsets[index] - (index == 0u ? 0u : leaf_offsets[index - 1u]);
}

__device__ inline std::uint32_t
start_depth_from_scanned_offsets(const std::uint32_t* node_offsets,
                                 const std::uint32_t* leaf_offsets,
                                 std::uint32_t index) {
    const std::uint32_t leaf_count =
        scanned_leaf_emit_count(leaf_offsets, index);
    if (leaf_count == 0u)
        return 0u;

    const std::uint32_t node_count =
        scanned_node_emit_count(node_offsets, index);
    return node_count == 0u ? kMaxDepth : kMaxDepth - node_count;
}

__device__ inline std::uint32_t
base_node_rank_from_scanned_offsets(const std::uint32_t* node_offsets,
                                    std::uint32_t index) {
    return index == 0u ? 0u : node_offsets[index - 1u];
}

__device__ inline std::uint32_t
base_leaf_rank_from_scanned_offsets(const std::uint32_t* leaf_offsets,
                                    std::uint32_t index) {
    return index == 0u ? 0u : leaf_offsets[index - 1u];
}

} // namespace recover_start_depth

template <> struct RankTraits<RecoverStartDepth> {
    template <class Workspace>
    __device__ static std::uint32_t start_depth(const Workspace& workspace,
                                                std::uint32_t index) {
        return recover_start_depth::start_depth_from_scanned_offsets(
            workspace.node_offsets, workspace.leaf_offsets, index);
    }

    template <class Workspace>
    __device__ static std::uint32_t
    node_base(const Workspace& workspace, std::uint32_t index, std::uint32_t) {
        return recover_start_depth::base_node_rank_from_scanned_offsets(
            workspace.node_offsets, index);
    }

    template <class Workspace>
    __device__ static std::uint32_t
    leaf_base(const Workspace& workspace, std::uint32_t index, std::uint32_t) {
        return recover_start_depth::base_leaf_rank_from_scanned_offsets(
            workspace.leaf_offsets, index);
    }

    template <class Workspace>
    __device__ static std::uint32_t
    emitted_node_rank(const Workspace& workspace, std::uint32_t leaf_index,
                      std::uint32_t depth) {
        const std::uint32_t start = start_depth(workspace, leaf_index);
        return node_base(workspace, leaf_index, start) + (depth - start);
    }

    template <class Workspace>
    __device__ static void record_start_depth(const Workspace&, std::uint32_t,
                                              std::uint32_t) {}
};

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail
