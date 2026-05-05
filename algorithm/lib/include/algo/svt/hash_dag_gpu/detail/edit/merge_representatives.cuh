#pragma once

#include <algo/cuda/acceleration_hash.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/node.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

template <std::uint32_t ItemWords, class Table>
__device__ inline std::uint32_t
merge_selected_item(Table &table, std::uint32_t src_lane,
                    const std::uint32_t *item, std::uint32_t *edit_status) {
  const std::uint32_t lane = lane_id();
  // Each representative item is merged with the persistent canonical table.
  // Existing equal items are reused; otherwise this batch publishes a new item.
  bool find_active = lane == src_lane;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;
  table.find(find_active, item, ptr, status);

  bool insert_active =
      lane == src_lane && status == algo::cuda::kAccelerationHashStatusNotFound;
  std::uint32_t insert_ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t insert_status = algo::cuda::kAccelerationHashStatusNotFound;
  table.insert_unique_unchecked(insert_active, item, insert_ptr, insert_status);

  std::uint32_t ref = kEmptyRef;
  if (lane == src_lane) {
    if (status == algo::cuda::kAccelerationHashStatusFound) {
      ref = make_table_ref(ItemWords, ptr);
    } else if (insert_status == algo::cuda::kAccelerationHashStatusInserted) {
      ref = make_table_ref(ItemWords, insert_ptr);
    } else {
      set_edit_status(edit_status, kEditStatusHashOverflow);
    }
  }
  return ref;
}

__global__ void merge_representatives_kernel(
    DeviceHashDagGpu dag, const EditSvoRequest *requests,
    const std::uint32_t *request_count, std::uint32_t depth,
    const NodeItem *node_items, const std::uint32_t *item_words,
    const std::uint32_t *representative_ids, std::uint32_t *canonical_refs,
    std::uint32_t *edit_status) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t lane = lane_id();
  const bool in_range = index < *request_count;
  const bool active = in_range && request_depth(requests[index]) == depth &&
                      is_request_non_uniform(node_items, index) &&
                      representative_ids[index] == index;
  std::uint32_t active_mask = __ballot_sync(
      algo::cuda::acceleration_hash::detail::kFullWarpMask, active);
  const std::uint32_t *item = in_range ? node_items[index].words : nullptr;
  const std::uint32_t words = in_range ? item_words[index] : 0u;

  // Hash-set operations are warp-cooperative, so the warp serializes the active
  // representative lanes and gives each selected item the whole warp.
  while (active_mask != 0u) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(active_mask) - 1);
    const std::uint32_t selected_words = __shfl_sync(
        algo::cuda::acceleration_hash::detail::kFullWarpMask, words, src_lane);
    std::uint32_t ref = kEmptyRef;
    switch (selected_words) {
    case 1:
      ref = merge_selected_item<1>(dag.table1, src_lane, item, edit_status);
      break;
    case 2:
      ref = merge_selected_item<2>(dag.table2, src_lane, item, edit_status);
      break;
    case 3:
      ref = merge_selected_item<3>(dag.table3, src_lane, item, edit_status);
      break;
    case 4:
      ref = merge_selected_item<4>(dag.table4, src_lane, item, edit_status);
      break;
    case 5:
      ref = merge_selected_item<5>(dag.table5, src_lane, item, edit_status);
      break;
    case 6:
      ref = merge_selected_item<6>(dag.table6, src_lane, item, edit_status);
      break;
    case 7:
      ref = merge_selected_item<7>(dag.table7, src_lane, item, edit_status);
      break;
    case 8:
      ref = merge_selected_item<8>(dag.table8, src_lane, item, edit_status);
      break;
    case 9:
      ref = merge_selected_item<9>(dag.table9, src_lane, item, edit_status);
      break;
    default:
      if (lane == src_lane)
        set_edit_status(edit_status, kEditStatusWorkspaceOverflow);
      break;
    }

    if (lane == src_lane)
      canonical_refs[index] = ref;
    active_mask &= active_mask - 1u;
  }
}

__global__ void copy_representative_refs_kernel(
    const EditSvoRequest *requests, const std::uint32_t *request_count,
    std::uint32_t depth, const NodeItem *node_items,
    const std::uint32_t *representative_ids, std::uint32_t *canonical_refs) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= *request_count)
    return;
  if (request_depth(requests[index]) != depth)
    return;
  if (!is_request_non_uniform(node_items, index))
    return;

  // Non-representative duplicates reuse the ref produced for their
  // representative request in merge_representatives_kernel.
  canonical_refs[index] = canonical_refs[representative_ids[index]];
}

} // namespace algo::svt::hash_dag_gpu::detail
