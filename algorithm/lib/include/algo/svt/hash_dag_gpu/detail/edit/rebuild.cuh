#pragma once

#include <algo/cuda/acceleration_hash.cuh>
#include <algo/svt/hash_dag_gpu/config.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/merge_representatives.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/node_items.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/scatter_parent.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/temp_dedup.cuh>
#include <algo/svt/hash_dag_gpu/detail/workspace/edit_batch.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

inline cudaError_t scatter_leaves_to_parents(std::uint32_t count,
                                             EditWorkspace &workspace,
                                             cudaStream_t stream) {
  const std::uint32_t max_requests = max_request_count(count);
  const std::uint32_t blocks = block_count(max_requests, kKernelBlockSize);
  scatter_level_to_parent_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.requests, workspace.request_count, kInnerDepth,
      workspace.canonical_refs, workspace.child_refs);
  return last_launch_status();
}

inline cudaError_t process_depth(DeviceHashDagGpu dag, std::uint32_t count,
                                 std::uint32_t depth, EditWorkspace &workspace,
                                 cudaStream_t stream) {
  // Rebuild all requests at one depth from their child_refs, deduplicate equal
  // node items within this batch, merge representatives into the persistent DAG
  // tables, then scatter rebuilt refs to their parents.
  const std::uint32_t max_requests = max_request_count(count);
  const std::uint32_t blocks = block_count(max_requests, kKernelBlockSize);

  build_level_node_items_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.requests, workspace.request_count, depth, workspace.child_refs,
      workspace.node_items, workspace.item_words, workspace.temp_keys,
      workspace.representative_ids, workspace.canonical_refs);
  cudaError_t status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  status = algo::cuda::initialize_acceleration_hash_map32(
      workspace.temp_dedup_map, stream);
  if (status != cudaSuccess)
    return status;

  temp_dedup_insert_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.temp_dedup_map, workspace.requests, workspace.request_count,
      depth, workspace.node_items, workspace.temp_keys, workspace.edit_status);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  temp_dedup_find_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.temp_dedup_map, workspace.requests, workspace.request_count,
      depth, workspace.node_items, workspace.temp_keys,
      workspace.representative_ids, workspace.edit_status);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  merge_representatives_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      dag, workspace.requests, workspace.request_count, depth,
      workspace.node_items, workspace.item_words, workspace.representative_ids,
      workspace.canonical_refs, workspace.edit_status);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  copy_representative_refs_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.requests, workspace.request_count, depth, workspace.node_items,
      workspace.representative_ids, workspace.canonical_refs);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  if (depth == 0u)
    return cudaSuccess;

  scatter_level_to_parent_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.requests, workspace.request_count, depth,
      workspace.canonical_refs, workspace.child_refs);
  return last_launch_status();
}

} // namespace algo::svt::hash_dag_gpu::detail
