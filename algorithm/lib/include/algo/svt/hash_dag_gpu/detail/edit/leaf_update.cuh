#pragma once

#include <algo/cuda/acceleration_hash.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/workspace/edit_batch.cuh>
#include <algo/svt/hash_dag_gpu/detail/node.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

__device__ inline std::uint32_t
merge_selected_leaf(DeviceHashDagGpu dag, std::uint32_t src_lane,
                    std::uint64_t mask, std::uint32_t *edit_status) {
  const std::uint32_t lane = lane_id();
  // One lane owns the leaf being inserted, but the hash table primitives are
  // warp-cooperative. Broadcast the selected mask so all lanes have the item.
  const std::uint64_t selected_mask = __shfl_sync(
      algo::cuda::acceleration_hash::detail::kFullWarpMask, mask, src_lane);
  std::uint32_t item[2] = {static_cast<std::uint32_t>(selected_mask),
                           static_cast<std::uint32_t>(selected_mask >> 32u)};

  bool find_active = lane == src_lane;
  std::uint32_t ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t status = algo::cuda::kAccelerationHashStatusNotFound;
  dag.table2.find(find_active, item, ptr, status);

  bool insert_active =
      lane == src_lane && status == algo::cuda::kAccelerationHashStatusNotFound;
  std::uint32_t insert_ptr = algo::cuda::kAccelerationHashNullPtr;
  std::uint32_t insert_status = algo::cuda::kAccelerationHashStatusNotFound;
  dag.table2.insert_unique_unchecked(insert_active, item, insert_ptr,
                                     insert_status);

  std::uint32_t ref = kEmptyRef;
  if (lane == src_lane) {
    if (status == algo::cuda::kAccelerationHashStatusFound) {
      ref = make_table_ref(2u, ptr);
    } else if (insert_status == algo::cuda::kAccelerationHashStatusInserted) {
      ref = make_table_ref(2u, insert_ptr);
    } else {
      set_edit_status(edit_status, kEditStatusHashOverflow);
    }
  }
  return ref;
}

__global__ void apply_leaf_masks_kernel(const EditSvoRequest *requests,
                                        const std::uint32_t *request_count,
                                        const LeafEditMask *leaf_masks,
                                        DeviceHashDagGpu dag, EditOp op,
                                        std::uint32_t *canonical_refs,
                                        std::uint32_t *edit_status) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const bool in_range = index < *request_count;
  const bool active = in_range && request_depth(requests[index]) == kInnerDepth;

  std::uint64_t leaf_mask = 0ull;
  if (active) {
    const EditSvoRequest request = requests[index];
    leaf_mask = read_leaf_mask(dag, canonical_refs[index]);
    // Place is an OR into the leaf mask; destroy clears the selected bits.
    const std::uint64_t edit_mask =
        leaf_masks[request.source_index].voxel_mask;
    if (op == EditOp::Place)
      leaf_mask |= edit_mask;
    else
      leaf_mask &= ~edit_mask;

    if (leaf_mask == 0ull) {
      canonical_refs[index] = kEmptyRef;
    } else if (leaf_mask == ~0ull) {
      canonical_refs[index] = kFullRef;
    }
  }

  std::uint32_t active_mask =
      __ballot_sync(algo::cuda::acceleration_hash::detail::kFullWarpMask,
                    active && leaf_mask != 0ull && leaf_mask != ~0ull);
  // Process non-uniform leaves one selected lane at a time so each table merge
  // can use the full warp-cooperative hash operation.
  while (active_mask != 0u) {
    const std::uint32_t src_lane =
        static_cast<std::uint32_t>(__ffs(active_mask) - 1);
    const std::uint32_t ref =
        merge_selected_leaf(dag, src_lane, leaf_mask, edit_status);
    if (lane_id() == src_lane)
      canonical_refs[index] = ref;
    active_mask &= active_mask - 1u;
  }
}

inline cudaError_t apply_leaf_masks(std::uint32_t count, EditOp op,
                                    DeviceHashDagGpu dag,
                                    EditWorkspace &workspace,
                                    cudaStream_t stream) {
  // Apply the reduced voxel masks to leaf refs first; parent nodes are rebuilt
  // from these canonical leaf refs in the bottom-up pass.
  const std::uint32_t max_requests = max_request_count(count);
  const std::uint32_t blocks = block_count(max_requests, kKernelBlockSize);
  apply_leaf_masks_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.requests, workspace.request_count, workspace.leaf_masks, dag,
      op, workspace.canonical_refs, workspace.edit_status);
  return last_launch_status();
}

} // namespace algo::svt::hash_dag_gpu::detail
