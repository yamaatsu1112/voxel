#pragma once

#include <algo/svt/hash_dag_gpu/detail/node.cuh>

#include <cstdint>

namespace algo::svt::hash_dag_gpu {

// Device query helper for kernels that need random voxel lookup. Traversal can
// terminate early on the empty/full sentinels because they summarize an entire
// subtree without a table lookup.
__device__ inline bool get_voxel(DeviceHashDagGpu dag, std::uint32_t x,
                                 std::uint32_t y, std::uint32_t z) {
  if (!detail::valid_world_coord(x) || !detail::valid_world_coord(y) ||
      !detail::valid_world_coord(z)) {
    return false;
  }

  std::uint32_t ref = *dag.root_ref;
  for (std::uint32_t depth = 0u; depth < kInnerDepth; ++depth) {
    if (detail::is_empty_ref(ref))
      return false;
    if (detail::is_full_ref(ref))
      return true;

    const std::uint32_t child =
        detail::child_index_for_voxel_level(x, y, z, depth);
    ref = detail::child_ref(dag, ref, child);
  }

  if (detail::is_empty_ref(ref))
    return false;
  if (detail::is_full_ref(ref))
    return true;

  const std::uint64_t mask = detail::read_leaf_mask(dag, ref);
  const std::uint32_t bit = detail::leaf_local_bit_index(x, y, z);
  return (mask & (1ull << bit)) != 0ull;
}

} // namespace algo::svt::hash_dag_gpu
