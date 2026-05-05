#pragma once

#include <algo/svt/cuda/config.cuh>

#include <cstdint>

namespace algo::svt::cuda {

// A single voxel-space edit. Coordinates are absolute world coordinates, not
// leaf-local coordinates.
struct VoxelEdit {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
};

// Aggregated edits for one leaf. leaf_key identifies the leaf path in the tree,
// and the two masks identify which of the 64 leaf voxels are affected.
struct LeafMask {
    std::uint32_t leaf_key;
    std::uint32_t voxel_data_low;
    std::uint32_t voxel_data_high;
};

} // namespace algo::svt::cuda
