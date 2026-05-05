#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/cuda/sort/sort.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

// Typed layout over caller-provided scratch memory for one edit batch. The
// pipeline reuses arrays across phases to avoid device allocations during
// edits.
struct EditWorkspace {
  // Raw edits packed as (leaf key, voxel bit), then sorted by leaf key.
  std::uint32_t *keys;
  std::uint64_t *voxel_bits;
  std::uint32_t *run_offsets;
  // One OR-reduced voxel mask per edited leaf.
  LeafEditMask *leaf_masks;
  std::uint32_t *leaf_count;

  // Per-depth request offsets and compacted requests for touched leaves and
  // ancestors.
  std::uint32_t *request_offsets;
  EditSvoRequest *requests;
  std::uint32_t *request_count;
  std::uint32_t *level_offsets;
  std::uint32_t *level_counts;

  // Per-request refs while rebuilding: old_refs is the original DAG state,
  // canonical_refs is the rebuilt node ref, and child_refs is the editable
  // 8-child array used to rebuild each parent.
  std::uint32_t *old_refs;
  std::uint32_t *canonical_refs;
  std::uint32_t *child_refs;

  // Temporary node payloads and deduplication data for one depth at a time.
  NodeItem *node_items;
  std::uint32_t *item_words;
  std::uint32_t *temp_keys;
  std::uint32_t *representative_ids;
  std::uint32_t *edit_status;

  TempDedupSlab *temp_dedup_slabs;
  std::uint32_t *temp_dedup_allocator_next;
  TempDedupMap temp_dedup_map;

  void *sort_workspace;
  std::size_t sort_workspace_size;
  void *scan_workspace;
  std::size_t scan_workspace_size;
};

template <class T> constexpr std::size_t align_up(std::size_t offset) {
  constexpr std::size_t kAlign = alignof(T);
  return (offset + kAlign - 1u) & ~(kAlign - 1u);
}

template <class T> T *pointer_at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<std::byte *>(base) + offset);
}

template <class T>
inline void reserve_array(void *workspace, std::size_t count,
                          std::size_t &offset, T *&out) {
  offset = align_up<T>(offset);
  out = pointer_at<T>(workspace, offset);
  offset += sizeof(T) * count;
}

template <class T>
inline void reserve_workspace(void *workspace, std::size_t size,
                              std::size_t &offset, void *&out) {
  offset = align_up<T>(offset);
  out = pointer_at<std::byte>(workspace, offset);
  offset += size;
}

inline std::uint32_t temp_bucket_count(std::uint32_t count) {
  return std::max(
      1u, ceil_div_u32(count, algo::cuda::kAccelerationHashSlotsPerSlab));
}

inline std::uint32_t temp_overflow_slab_count(std::uint32_t count) {
  return std::max(
      1u, ceil_div_u32(count, algo::cuda::kAccelerationHashSlotsPerSlab));
}

inline std::uint32_t temp_slab_count(std::uint32_t count) {
  return temp_bucket_count(count) + temp_overflow_slab_count(count);
}

inline std::size_t edit_workspace_size(std::uint32_t count) {
  // Keep this layout in exact lockstep with create_edit_workspace(). The
  // explicit alignment mirrors reserve_array/reserve_workspace below.
  const std::uint32_t max_requests = max_request_count(count);
  const std::uint32_t request_offsets = request_offset_count(count);
  const std::uint32_t temp_slabs = temp_slab_count(count);
  const std::uint32_t scan_count = std::max<std::uint32_t>(count, 1u);

  std::size_t offset = 0;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * count;
  offset = align_up<std::uint64_t>(offset);
  offset += sizeof(std::uint64_t) * count;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * count;
  offset = align_up<LeafEditMask>(offset);
  offset += sizeof(LeafEditMask) * count;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);

  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * request_offsets;
  offset = align_up<EditSvoRequest>(offset);
  offset += sizeof(EditSvoRequest) * max_requests;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * (kEditDepthCount + 1u);
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * kEditDepthCount;

  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * max_requests;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * max_requests;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * max_requests * kGroupSize;

  offset = align_up<NodeItem>(offset);
  offset += sizeof(NodeItem) * max_requests;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * max_requests;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * max_requests * kTempDedupKeyWords;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t) * max_requests;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);

  offset = align_up<TempDedupSlab>(offset);
  offset += sizeof(TempDedupSlab) * temp_slabs;
  offset = align_up<std::uint32_t>(offset);
  offset += sizeof(std::uint32_t);

  offset = align_up<std::max_align_t>(offset);
  offset +=
      algo::cuda::sort::required_pairs_workspace_size<std::uint32_t,
                                                      std::uint64_t>(count);

  offset = align_up<std::max_align_t>(offset);
  offset += algo::cuda::scan::required_workspace_size_fused<std::uint32_t>(
      scan_count);
  return offset;
}

inline EditWorkspace create_edit_workspace(void *workspace,
                                           std::uint32_t count) {
  // Carve the untyped byte buffer into all arrays used by the edit pipeline.
  // No ownership is taken; callers must pass at least edit_workspace_size().
  const std::uint32_t max_requests = max_request_count(count);
  const std::uint32_t request_offsets = request_offset_count(count);
  const std::uint32_t scan_count = std::max<std::uint32_t>(count, 1u);
  const std::uint32_t buckets = temp_bucket_count(count);
  const std::uint32_t temp_slabs = temp_slab_count(count);

  EditWorkspace result{};
  std::size_t offset = 0;
  reserve_array(workspace, count, offset, result.keys);
  reserve_array(workspace, count, offset, result.voxel_bits);
  reserve_array(workspace, count, offset, result.run_offsets);
  reserve_array(workspace, count, offset, result.leaf_masks);
  reserve_array(workspace, 1, offset, result.leaf_count);

  reserve_array(workspace, request_offsets, offset, result.request_offsets);
  reserve_array(workspace, max_requests, offset, result.requests);
  reserve_array(workspace, 1, offset, result.request_count);
  reserve_array(workspace, kEditDepthCount + 1u, offset, result.level_offsets);
  reserve_array(workspace, kEditDepthCount, offset, result.level_counts);

  reserve_array(workspace, max_requests, offset, result.old_refs);
  reserve_array(workspace, max_requests, offset, result.canonical_refs);
  reserve_array(workspace, max_requests * kGroupSize, offset,
                result.child_refs);

  reserve_array(workspace, max_requests, offset, result.node_items);
  reserve_array(workspace, max_requests, offset, result.item_words);
  reserve_array(workspace, max_requests * kTempDedupKeyWords, offset,
                result.temp_keys);
  reserve_array(workspace, max_requests, offset, result.representative_ids);
  reserve_array(workspace, 1, offset, result.edit_status);

  reserve_array(workspace, temp_slabs, offset, result.temp_dedup_slabs);
  reserve_array(workspace, 1, offset, result.temp_dedup_allocator_next);
  result.temp_dedup_map = TempDedupMap{
      result.temp_dedup_slabs, buckets, temp_slabs,
      DeviceHashDagGpu::Allocator::Device{result.temp_dedup_allocator_next}};

  result.sort_workspace_size =
      algo::cuda::sort::required_pairs_workspace_size<std::uint32_t,
                                                      std::uint64_t>(count);
  reserve_workspace<std::max_align_t>(workspace, result.sort_workspace_size,
                                      offset, result.sort_workspace);

  result.scan_workspace_size =
      algo::cuda::scan::required_workspace_size_fused<std::uint32_t>(
          scan_count);
  reserve_workspace<std::max_align_t>(workspace, result.scan_workspace_size,
                                      offset, result.scan_workspace);
  return result;
}

} // namespace algo::svt::hash_dag_gpu::detail
