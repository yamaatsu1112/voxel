#pragma once

#include <algo/cuda/acceleration_hash.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/node.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

__global__ void temp_dedup_insert_kernel(TempDedupMap table,
                                         const EditSvoRequest *requests,
                                         const std::uint32_t *request_count,
                                         std::uint32_t depth,
                                         const NodeItem *node_items,
                                         const std::uint32_t *temp_keys,
                                         std::uint32_t *edit_status) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = index < *request_count &&
                request_depth(requests[index]) == depth &&
                is_request_non_uniform(node_items, index);
  const std::uint32_t *key =
      active ? temp_keys + index * kTempDedupKeyWords : nullptr;
  const std::uint32_t value = active ? index : 0u;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;

  // The temporary map is per-depth and maps node item -> first request index.
  // Later equal requests copy that representative's canonical ref instead of
  // merging the same item into the persistent DAG repeatedly.
  table.insert_unique_unchecked(active, key, value, ptr, status);
  if (active && status == algo::cuda::kAccelerationHashStatusOverflow)
    set_edit_status(edit_status, kEditStatusHashOverflow);
}

__global__ void temp_dedup_find_kernel(
    TempDedupMap table, const EditSvoRequest *requests,
    const std::uint32_t *request_count, std::uint32_t depth,
    const NodeItem *node_items, const std::uint32_t *temp_keys,
    std::uint32_t *representative_ids, std::uint32_t *edit_status) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = index < *request_count &&
                request_depth(requests[index]) == depth &&
                is_request_non_uniform(node_items, index);
  const std::uint32_t *key =
      active ? temp_keys + index * kTempDedupKeyWords : nullptr;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t value = 0u;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;

  table.find(active, key, ptr, value, status);
  if (active) {
    // The value returned by the map is the representative request chosen during
    // insertion. If lookup fails, mark an overflow-style error so the root is
    // not published.
    if (status == algo::cuda::kAccelerationHashStatusFound) {
      representative_ids[index] = value;
    } else {
      representative_ids[index] = index;
      set_edit_status(edit_status, kEditStatusHashOverflow);
    }
  }
}

} // namespace algo::svt::hash_dag_gpu::detail
