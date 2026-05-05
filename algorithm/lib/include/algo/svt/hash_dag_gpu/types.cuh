#pragma once

#include <algo/svt/hash_dag_gpu/config.cuh>

#include <cstdint>

namespace algo::svt::hash_dag_gpu {

// Public edit coordinate. Coordinates outside [0, kWorldVoxelCount) are ignored
// by the edit pipeline and read as empty by queries.
struct VoxelEdit {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
};

// A batch of voxel edits is first reduced to one mask per leaf key. leaf_key is
// the path from the root to the leaf, packed as 3-bit child indices.
struct LeafEditMask {
  std::uint32_t leaf_key;
  std::uint64_t voxel_mask;
};

// Canonical internal-node payload. words[0] is the packed header; following
// words are only the non-uniform child refs selected by child_mask.
struct NodeItem {
  std::uint32_t words[kMaxItemWords];
};

enum class EditOp {
  Place,
  Destroy,
};

} // namespace algo::svt::hash_dag_gpu
