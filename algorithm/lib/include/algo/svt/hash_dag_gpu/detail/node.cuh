#pragma once

#include <algo/cuda/acceleration_hash.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

// Shared encoding helpers for the DAG. Most are host/device so tests or host
// setup code can reason about the same packed representation used by kernels.
__host__ __device__ inline bool valid_world_coord(std::uint32_t coord) {
  return coord < kWorldVoxelCount;
}

__host__ __device__ inline bool is_empty_ref(std::uint32_t ref) {
  return ref == kEmptyRef;
}

__host__ __device__ inline bool is_full_ref(std::uint32_t ref) {
  return ref == kFullRef;
}

__host__ __device__ inline bool is_table_ref(std::uint32_t ref) {
  return ref > kFullRef;
}

__host__ __device__ inline std::uint32_t
make_table_ref(std::uint32_t item_words, std::uint32_t ptr) {
  return (item_words << kRefItemWordsShift) | (ptr & kRefPtrMask);
}

__host__ __device__ inline std::uint32_t ref_item_words(std::uint32_t ref) {
  return ref >> kRefItemWordsShift;
}

__host__ __device__ inline std::uint32_t ref_hash_ptr(std::uint32_t ref) {
  return ref & kRefPtrMask;
}

__host__ __device__ inline std::uint32_t make_header(std::uint32_t child_mask,
                                                     std::uint32_t fill_mask) {
  return ((child_mask & kNodeMask) << kChildMaskShift) |
         ((fill_mask & kNodeMask) << kFillMaskShift);
}

__host__ __device__ inline std::uint32_t
header_child_mask(std::uint32_t header) {
  return (header >> kChildMaskShift) & kNodeMask;
}

__host__ __device__ inline std::uint32_t
header_fill_mask(std::uint32_t header) {
  return (header >> kFillMaskShift) & kNodeMask;
}

__host__ __device__ inline std::uint32_t popcount_u32(std::uint32_t value) {
  std::uint32_t count = 0;
  while (value != 0u) {
    value &= value - 1u;
    ++count;
  }
  return count;
}

__host__ __device__ inline std::uint32_t
child_index_for_voxel_level(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                            std::uint32_t level) {
  // Each inner level consumes one bit from x/y/z, starting at the most
  // significant world bit. The three bits form the octree child index.
  const std::uint32_t shift = kWorldVoxelCountExp - level - 1u;
  return ((x >> shift) & 1u) | (((y >> shift) & 1u) << 1u) |
         (((z >> shift) & 1u) << 2u);
}

__host__ __device__ inline std::uint32_t
make_leaf_key(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  // The key is only the inner-node path. The remaining low coordinate bits are
  // encoded separately as a bit inside the 4^3 leaf mask.
  std::uint32_t key = 0u;
  for (std::uint32_t depth = 0u; depth < kInnerDepth; ++depth)
    key = (key << kGroupSizeExp) | child_index_for_voxel_level(x, y, z, depth);
  return key;
}

__host__ __device__ inline std::uint32_t
leaf_local_bit_index(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  const std::uint32_t local_x = x & (kLeafVoxelCount - 1u);
  const std::uint32_t local_y = y & (kLeafVoxelCount - 1u);
  const std::uint32_t local_z = z & (kLeafVoxelCount - 1u);
  return local_x | (local_y << kLeafVoxelCountExp) |
         (local_z << (kLeafVoxelCountExp * 2u));
}

#ifdef __CUDACC__

__device__ inline std::uint32_t lane_id() {
  return static_cast<std::uint32_t>(threadIdx.x) &
         (algo::cuda::kSlabHashWarpSize - 1u);
}

template <std::uint32_t ItemWords>
__device__ inline bool
read_table_item(const algo::cuda::DeviceAccelerationHashSet32<
                    ItemWords, DeviceHashDagGpu::Allocator> &table,
                std::uint32_t ptr, std::uint32_t (&out)[kMaxItemWords]) {
  std::uint32_t item[ItemWords]{};
  if (!table.read_item(ptr, item))
    return false;
  for (std::uint32_t i = 0; i < ItemWords; ++i)
    out[i] = item[i];
  return true;
}

__device__ inline bool read_node_item(DeviceHashDagGpu dag, std::uint32_t ref,
                                      std::uint32_t (&out)[kMaxItemWords]) {
  if (!is_table_ref(ref))
    return false;

  const std::uint32_t ptr = ref_hash_ptr(ref);
  // The ref stores the item width, which selects the matching fixed-width hash
  // table. Fixed-width tables keep the GPU hash implementation simple.
  switch (ref_item_words(ref)) {
  case 1:
    return read_table_item(dag.table1, ptr, out);
  case 2:
    return read_table_item(dag.table2, ptr, out);
  case 3:
    return read_table_item(dag.table3, ptr, out);
  case 4:
    return read_table_item(dag.table4, ptr, out);
  case 5:
    return read_table_item(dag.table5, ptr, out);
  case 6:
    return read_table_item(dag.table6, ptr, out);
  case 7:
    return read_table_item(dag.table7, ptr, out);
  case 8:
    return read_table_item(dag.table8, ptr, out);
  case 9:
    return read_table_item(dag.table9, ptr, out);
  default:
    return false;
  }
}

__device__ inline std::uint32_t
child_ref_from_item(const std::uint32_t (&item)[kMaxItemWords],
                    std::uint32_t child_index) {
  const std::uint32_t child_mask = header_child_mask(item[0]);
  const std::uint32_t fill_mask = header_fill_mask(item[0]);
  const std::uint32_t child_bit = 1u << child_index;
  if ((child_mask & child_bit) == 0u)
    return (fill_mask & child_bit) != 0u ? kFullRef : kEmptyRef;

  // Child refs are stored densely after the header. popcount before this bit
  // gives the dense ordinal for child_index.
  const std::uint32_t ordinal = popcount_u32(child_mask & (child_bit - 1u));
  return item[1u + ordinal];
}

__device__ inline std::uint32_t child_ref(DeviceHashDagGpu dag,
                                          std::uint32_t parent_ref,
                                          std::uint32_t child_index) {
  if (is_empty_ref(parent_ref))
    return kEmptyRef;
  if (is_full_ref(parent_ref))
    return kFullRef;

  std::uint32_t item[kMaxItemWords]{};
  if (!read_node_item(dag, parent_ref, item))
    return kEmptyRef;
  return child_ref_from_item(item, child_index);
}

template <std::uint32_t ItemWords>
__device__ inline std::uint32_t find_or_insert_item(
    algo::cuda::DeviceAccelerationHashSet32<ItemWords,
                                            DeviceHashDagGpu::Allocator> &table,
    const std::uint32_t *item) {
  // AccelerationHash operations are warp-cooperative. Only lane 0 issues the
  // logical lookup/insert for this helper, then broadcasts the resulting ref.
  const std::uint32_t lane = lane_id();
  bool active = lane == 0u;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;
  table.find(active, item, ptr, status);

  bool insert_active =
      lane == 0u && status == algo::cuda::kAccelerationHashStatusNotFound;
  std::uint32_t insert_ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t insert_status = algo::cuda::kAccelerationHashStatusNotFound;
  table.insert_unique_unchecked(insert_active, item, insert_ptr, insert_status);
  if (lane == 0u &&
      insert_status == algo::cuda::kAccelerationHashStatusInserted) {
    ptr = insert_ptr;
    status = insert_status;
  }

  std::uint32_t ref = kEmptyRef;
  if (lane == 0u && (status == algo::cuda::kAccelerationHashStatusFound ||
                     status == algo::cuda::kAccelerationHashStatusInserted)) {
    ref = make_table_ref(ItemWords, ptr);
  }
  return __shfl_sync(algo::cuda::acceleration_hash::detail::kFullWarpMask, ref,
                     0);
}

__device__ inline std::uint32_t canonicalize_item(DeviceHashDagGpu dag,
                                                  const std::uint32_t *item,
                                                  std::uint32_t item_words) {
  // Collapse uniform nodes to sentinels before touching hash tables. This keeps
  // large empty/full regions cheap and maximizes structural sharing.
  const std::uint32_t child_mask = header_child_mask(item[0]);
  const std::uint32_t fill_mask = header_fill_mask(item[0]);
  if (child_mask == 0u) {
    if (fill_mask == 0u)
      return kEmptyRef;
    if (fill_mask == kNodeMask)
      return kFullRef;
  }

  switch (item_words) {
  case 1:
    return find_or_insert_item(dag.table1, item);
  case 2:
    return find_or_insert_item(dag.table2, item);
  case 3:
    return find_or_insert_item(dag.table3, item);
  case 4:
    return find_or_insert_item(dag.table4, item);
  case 5:
    return find_or_insert_item(dag.table5, item);
  case 6:
    return find_or_insert_item(dag.table6, item);
  case 7:
    return find_or_insert_item(dag.table7, item);
  case 8:
    return find_or_insert_item(dag.table8, item);
  case 9:
    return find_or_insert_item(dag.table9, item);
  default:
    return kEmptyRef;
  }
}

__device__ inline std::uint64_t read_leaf_mask(DeviceHashDagGpu dag,
                                               std::uint32_t ref) {
  // Leaves are stored in table2 as two 32-bit words. Empty/full sentinels map to
  // the all-zero/all-one masks used by edit and query code.
  if (is_empty_ref(ref))
    return 0ull;
  if (is_full_ref(ref))
    return ~0ull;
  if (!is_table_ref(ref) || ref_item_words(ref) != 2u)
    return 0ull;

  std::uint32_t item[kMaxItemWords]{};
  if (!read_table_item(dag.table2, ref_hash_ptr(ref), item))
    return 0ull;
  return static_cast<std::uint64_t>(item[0]) |
         (static_cast<std::uint64_t>(item[1]) << 32u);
}

__device__ inline std::uint32_t
build_item_from_children(const std::uint32_t (&children)[kGroupSize],
                         std::uint32_t (&item)[kMaxItemWords]) {
  // Convert eight child refs into the canonical compact node item. Empty
  // children are implicit, full children set fill_mask, and only table-backed
  // children are appended as payload words.
  std::uint32_t child_mask = 0u;
  std::uint32_t fill_mask = 0u;
  std::uint32_t item_words = 1u;

  for (std::uint32_t child = 0u; child < kGroupSize; ++child) {
    const std::uint32_t bit = 1u << child;
    const std::uint32_t ref = children[child];
    if (is_full_ref(ref)) {
      fill_mask |= bit;
    } else if (is_table_ref(ref)) {
      child_mask |= bit;
      item[item_words++] = ref;
    }
  }

  item[0] = make_header(child_mask, fill_mask);
  return item_words;
}

#endif

} // namespace algo::svt::hash_dag_gpu::detail
