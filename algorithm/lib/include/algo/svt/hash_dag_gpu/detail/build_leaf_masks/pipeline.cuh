#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/cuda/sort/sort.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/workspace/edit_batch.cuh>
#include <algo/svt/hash_dag_gpu/detail/node.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

__global__ void pack_leaf_edit_records_kernel(const VoxelEdit *edits,
                                              std::uint32_t count,
                                              std::uint32_t *keys,
                                              std::uint64_t *bits) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const VoxelEdit edit = edits[index];
  // Invalid coordinates are converted into a sentinel key so the later sort and
  // reduction can ignore them without branching on the original edit array.
  if (!valid_world_coord(edit.x) || !valid_world_coord(edit.y) ||
      !valid_world_coord(edit.z)) {
    keys[index] = kInvalidNodeKey;
    bits[index] = 0u;
    return;
  }

  keys[index] = make_leaf_key(edit.x, edit.y, edit.z);
  bits[index] = 1ull << leaf_local_bit_index(edit.x, edit.y, edit.z);
}

__global__ void mark_leaf_run_offsets_kernel(const std::uint32_t *keys,
                                             std::uint32_t *run_offsets,
                                             std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const bool valid = keys[index] != kInvalidNodeKey;
  // Sort groups all edits for the same leaf into one contiguous run. Mark only
  // the first element of each valid run so the following exclusive scan turns
  // these flags into compact output indices for one LeafEditMask per leaf.
  const bool run_start =
      valid && (index == 0u || keys[index - 1u] != keys[index]);
  run_offsets[index] = run_start ? 1u : 0u;
}

__global__ void
reduce_leaf_runs_kernel(const std::uint32_t *keys, const std::uint64_t *bits,
                        const std::uint32_t *run_offsets, std::uint32_t count,
                        LeafEditMask *leaf_masks, std::uint32_t *leaf_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const bool valid = keys[index] != kInvalidNodeKey;
  const bool run_start =
      valid && (index == 0u || keys[index - 1u] != keys[index]);
  if (run_start) {
    // Runs are expected to be short for edit batches. This local linear reduce
    // avoids another temporary array just to OR duplicate leaf bits.
    std::uint64_t mask = 0ull;
    for (std::uint32_t i = index; i < count && keys[i] == keys[index]; ++i)
      mask |= bits[i];
    leaf_masks[run_offsets[index]] = LeafEditMask{keys[index], mask};
  }

  if (index == count - 1u)
    *leaf_count = run_offsets[index] + (run_start ? 1u : 0u);
}

// Convert arbitrary voxel edits into one bit mask per touched leaf. Sorting by
// leaf key makes duplicate edits and multiple voxels in the same leaf collapse
// before any DAG traversal happens.
inline cudaError_t build_leaf_masks(const VoxelEdit *edits, std::uint32_t count,
                                    EditWorkspace &workspace,
                                    cudaStream_t stream) {
  if (count == 0u)
    return cudaSuccess;

  const std::uint32_t blocks = block_count(count, kKernelBlockSize);
  pack_leaf_edit_records_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      edits, count, workspace.keys, workspace.voxel_bits);
  cudaError_t status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  status = algo::cuda::sort::sort_pairs(workspace.keys, workspace.voxel_bits,
                                        count, workspace.sort_workspace,
                                        workspace.sort_workspace_size, stream);
  if (status != cudaSuccess)
    return status;

  mark_leaf_run_offsets_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.keys, workspace.run_offsets, count);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  status = algo::cuda::scan::exclusive_sum(
      workspace.run_offsets, count, workspace.scan_workspace,
      workspace.scan_workspace_size, stream);
  if (status != cudaSuccess)
    return status;

  reduce_leaf_runs_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      workspace.keys, workspace.voxel_bits, workspace.run_offsets, count,
      workspace.leaf_masks, workspace.leaf_count);
  return last_launch_status();
}

} // namespace algo::svt::hash_dag_gpu::detail
