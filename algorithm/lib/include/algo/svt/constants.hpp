#pragma once

#include <cstdint>
#include <stdexcept>

namespace algo::svt {

constexpr float VOXEL_SIZE_INV = 16.0f;
constexpr uint32_t LEAF_VOXEL_COUNT_EXP = 2;
constexpr uint32_t LEAF_VOXEL_COUNT = 1u << LEAF_VOXEL_COUNT_EXP;

constexpr uint32_t DEFAULT_MAX_DEPTH = 10;

namespace detail {

[[nodiscard]] constexpr uint32_t depth_for_world_size(
    uint32_t requested_world_size, uint32_t terminal_world_size,
    uint32_t bits_per_depth, uint32_t min_depth, uint32_t max_depth) {
    if (requested_world_size == 0) {
        throw std::out_of_range("requested world size must be greater than zero");
    }

    uint32_t depth = min_depth;
    uint64_t world_size =
        uint64_t{terminal_world_size} << (bits_per_depth * min_depth);
    while (world_size < requested_world_size) {
        if (depth >= max_depth) {
            throw std::out_of_range("requested world size out of supported range");
        }
        world_size <<= bits_per_depth;
        ++depth;
    }
    return depth;
}

} // namespace detail

} // namespace algo::svt
