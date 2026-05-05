#pragma once

#include <algo/svt/cuda/config.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

// Pack a leaf coordinate into the sequence of child indices needed to reach it.
// The top-level child occupies the highest bits so normal integer sorting groups
// leaves by shared tree prefixes.
__host__ __device__ inline std::uint32_t make_leaf_key(std::uint32_t leaf_x,
                                                       std::uint32_t leaf_y,
                                                       std::uint32_t leaf_z) {
    std::uint32_t key = 0u;
    for (std::uint32_t depth = 0u; depth < kMaxDepth; ++depth) {
        const std::uint32_t bit = kMaxDepth - depth - 1u;
        const std::uint32_t child_index =
            ((leaf_x >> bit) & 1u) | (((leaf_y >> bit) & 1u) << 1u) |
            (((leaf_z >> bit) & 1u) << 2u);
        key = (key << kGroupSizeExp) | child_index;
    }
    return key;
}

__host__ __device__ inline std::uint32_t request_key_for_child(
    std::uint32_t node_index, std::uint32_t child_index) {
    return (node_index << kGroupSizeExp) | child_index;
}

__host__ __device__ inline std::uint32_t request_key_node_index(
    std::uint32_t request_key) {
    return request_key >> kGroupSizeExp;
}

__host__ __device__ inline std::uint32_t request_key_child_index(
    std::uint32_t request_key) {
    return request_key & (kGroupSize - 1u);
}

__device__ inline bool has_mask_bit(std::uint32_t low, std::uint32_t high,
                                    std::uint32_t index) {
    const std::uint32_t bit = 1u << (index & 31u);
    return index < 32u ? ((low & bit) != 0u) : ((high & bit) != 0u);
}

__device__ inline std::uint32_t child_index_for_leaf_key(std::uint32_t key,
                                                         std::uint32_t depth) {
    const std::uint32_t shift = (kMaxDepth - depth - 1u) * kGroupSizeExp;
    return (key >> shift) & (kGroupSize - 1u);
}

__host__ __device__ inline std::uint32_t prefix_for_leaf_key(
    std::uint32_t key, std::uint32_t depth) {
    const std::uint32_t shift = (kMaxDepth - depth) * kGroupSizeExp;
    return key >> shift;
}

__device__ inline std::uint32_t child_index_for_voxel(
    std::uint32_t x, std::uint32_t y, std::uint32_t z,
    std::uint32_t depth) {
    // Skip the leaf-local bits, then read the branch bits for this depth.
    const std::uint32_t shift =
        (kMaxDepth - depth - 1u) * kBranchFactorExp + kLeafVoxelCountExp;
    const std::uint32_t mask = (1u << kBranchFactorExp) - 1u;
    const std::uint32_t x_bits = (x >> shift) & mask;
    const std::uint32_t y_bits = (y >> shift) & mask;
    const std::uint32_t z_bits = (z >> shift) & mask;
    return x_bits | (y_bits << kBranchFactorExp) |
           (z_bits << (kBranchFactorExp * 2u));
}

__device__ inline std::uint32_t leaf_offset_for_voxel(
    std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    // The low coordinate bits address one voxel inside the 4^3 leaf payload.
    const std::uint32_t mask = (1u << kLeafVoxelCountExp) - 1u;
    const std::uint32_t x_offset = x & mask;
    const std::uint32_t y_offset = y & mask;
    const std::uint32_t z_offset = z & mask;
    return x_offset | (y_offset << kLeafVoxelCountExp) |
           (z_offset << (kLeafVoxelCountExp * 2u));
}

__device__ inline bool valid_world_coord(std::uint32_t coord) {
    return coord < kWorldVoxelCount;
}

} // namespace algo::svt::cuda::detail

namespace algo::svt::cuda {

inline std::uint32_t make_leaf_key(std::uint32_t leaf_x,
                                   std::uint32_t leaf_y,
                                   std::uint32_t leaf_z) {
    return detail::make_leaf_key(leaf_x, leaf_y, leaf_z);
}

} // namespace algo::svt::cuda
