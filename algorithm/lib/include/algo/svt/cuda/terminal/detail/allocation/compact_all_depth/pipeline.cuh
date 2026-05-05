#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/allocation/batch_kernels.cuh>
#include <algo/svt/cuda/detail/allocation/request_compaction.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/detail/node_ops.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/compact_all_depth/materialize_helpers.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/compact_all_depth/policy.cuh>
#include <algo/svt/cuda/terminal/detail/common.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {
namespace allocation::terminal_compact_all_depth {

__device__ inline std::uint32_t
request_target_depth(const TerminalRequest &request) {
  if (terminal_request_is_brick(request))
    return kMaxDepth;
  const CellWriteRequest cell = terminal_request_cell(request);
  return cell.level == 0u ? 0u : cell.level - 1u;
}

__device__ inline std::uint32_t lcp_depth(std::uint32_t lhs,
                                          std::uint32_t rhs) {
  std::uint32_t depth = 0u;
  for (; depth < kMaxDepth; ++depth) {
    if (child_index_for_leaf_key(lhs, depth) !=
        child_index_for_leaf_key(rhs, depth)) {
      break;
    }
  }
  return depth;
}

__device__ inline std::uint32_t
terminal_node_emit_count(std::uint32_t start_depth,
                         std::uint32_t target_depth) {
  if (start_depth == 0u || start_depth == kMaxDepth)
    return 0u;
  const std::uint32_t node_target =
      target_depth < kMaxDepth ? target_depth : kMaxDepth - 1u;
  return start_depth <= node_target ? node_target - start_depth + 1u : 0u;
}

__device__ inline std::uint32_t
terminal_leaf_emit_count(const TerminalRequest &request,
                         std::uint32_t start_depth) {
  return terminal_request_is_brick(request) && start_depth != 0u ? 1u : 0u;
}

__device__ inline std::uint32_t
request_start_depth(const TerminalRequest *requests, std::uint32_t index,
                    std::uint32_t leaf_key, std::uint32_t first_missing_depth) {
  if (first_missing_depth == 0u)
    return 0u;

  std::uint32_t unique_depth = 1u;
  if (index != 0u) {
    const TerminalRequest previous = requests[index - 1u];
    const std::uint32_t previous_leaf_key =
        terminal_request_representative_leaf_key(previous);
    const std::uint32_t previous_lcp = lcp_depth(previous_leaf_key, leaf_key);
    const std::uint32_t previous_target_depth = request_target_depth(previous);
    const std::uint32_t previous_cover_depth =
        previous_lcp < previous_target_depth ? previous_lcp
                                             : previous_target_depth;
    unique_depth = previous_cover_depth + 1u;
  }
  return first_missing_depth > unique_depth ? first_missing_depth
                                            : unique_depth;
}

__device__ inline std::uint32_t scanned_emit_count(const std::uint32_t *offsets,
                                                   std::uint32_t index) {
  return offsets[index] - (index == 0u ? 0u : offsets[index - 1u]);
}

__device__ inline std::uint32_t
recovered_start_depth(const TerminalRequest *requests,
                      const TerminalCompactAllocationWorkspace &workspace,
                      std::uint32_t index) {
  const std::uint32_t node_count =
      scanned_emit_count(workspace.materialize_node_offsets, index);
  if (node_count != 0u) {
    const std::uint32_t target_depth =
        workspace.materialize_target_depths[index];
    const std::uint32_t node_target =
        target_depth < kMaxDepth ? target_depth : kMaxDepth - 1u;
    return node_target - node_count + 1u;
  }
  if (scanned_emit_count(workspace.materialize_leaf_offsets, index) != 0u)
    return kMaxDepth;
  return 0u;
}

__device__ inline bool node_slot_emits(const TerminalRequest *requests,
                                       const TerminalCompactAllocationWorkspace &workspace,
                                       std::uint32_t index,
                                       std::uint32_t depth) {
  const std::uint32_t start_depth =
      recovered_start_depth(requests, workspace, index);
  if (start_depth == 0u || depth < start_depth || depth == kMaxDepth)
    return false;

  return depth <= workspace.materialize_target_depths[index];
}

__device__ inline bool leaf_slot_emits(const TerminalRequest *requests,
                                       const TerminalCompactAllocationWorkspace &workspace,
                                       std::uint32_t index) {
  return terminal_request_is_brick(requests[index]) &&
         scanned_emit_count(workspace.materialize_leaf_offsets, index) != 0u;
}

__device__ inline std::uint32_t
node_base(const TerminalCompactAllocationWorkspace &workspace, std::uint32_t index) {
  return index == 0u ? 0u : workspace.materialize_node_offsets[index - 1u];
}

__device__ inline std::uint32_t
leaf_base(const TerminalCompactAllocationWorkspace &workspace, std::uint32_t index) {
  return index == 0u ? 0u : workspace.materialize_leaf_offsets[index - 1u];
}

__device__ inline std::uint32_t
emitted_node_rank(const TerminalRequest *requests,
                  const TerminalCompactAllocationWorkspace &workspace, std::uint32_t index,
                  std::uint32_t depth) {
  const std::uint32_t start_depth =
      recovered_start_depth(requests, workspace, index);
  return node_base(workspace, index) + (depth - start_depth);
}

__device__ inline std::uint32_t
lower_bound_request_key(const TerminalRequest *requests, std::uint32_t count,
                        std::uint32_t key) {
  std::uint32_t first = 0u;
  std::uint32_t remaining = count;
  while (remaining != 0u) {
    const std::uint32_t step = remaining >> 1u;
    const std::uint32_t mid = first + step;
    if (terminal_request_representative_leaf_key(requests[mid]) < key) {
      first = mid + 1u;
      remaining -= step + 1u;
    } else {
      remaining = step;
    }
  }
  return first;
}

__device__ inline std::uint32_t
find_parent_owner(const TerminalRequest *requests,
                  const TerminalCompactAllocationWorkspace &workspace,
                  std::uint32_t request_count, std::uint32_t leaf_key,
                  std::uint32_t parent_depth) {
  const std::uint32_t parent_prefix =
      prefix_for_leaf_key(leaf_key, parent_depth);
  const std::uint32_t shift = (kMaxDepth - parent_depth) * kGroupSizeExp;
  const std::uint32_t first_key = parent_prefix << shift;
  const std::uint32_t owner =
      lower_bound_request_key(requests, request_count, first_key);
  if (owner >= request_count)
    return request_count;

  const std::uint32_t owner_key =
      terminal_request_representative_leaf_key(requests[owner]);
  if (prefix_for_leaf_key(owner_key, parent_depth) != parent_prefix)
    return request_count;
  if (!node_slot_emits(requests, workspace, owner, parent_depth))
    return request_count;
  return owner;
}

__global__ void collect_counts_kernel(DeviceGpuSvo svo,
                                      const TerminalRequest *requests,
                                      std::uint32_t request_count,
                                      TerminalCompactAllocationWorkspace workspace) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= request_count)
    return;

  const TerminalRequest request = requests[index];
  const std::uint32_t target_depth = request_target_depth(request);
  const std::uint32_t leaf_key =
      terminal_request_representative_leaf_key(request);
  std::uint32_t first_missing_depth = 0u;
  std::uint32_t existing_parent_index = kRootNodeIndex;
  std::uint32_t filled = 0u;

  std::uint32_t node_index = kRootNodeIndex;
  for (std::uint32_t depth = 1u; depth <= target_depth; ++depth) {
    const std::uint32_t child_index =
        child_index_for_leaf_key(leaf_key, depth - 1u);
    const GpuSvoNode node = svo.nodes[node_index];
    if (!node_has_child(node, child_index)) {
      first_missing_depth = depth;
      existing_parent_index = node_index;
      filled = node_is_filled(node, child_index) ? 1u : 0u;
      break;
    }
    node_index = node_child_index(node, child_index);
  }

  workspace.materialize_target_depths[index] = target_depth;
  workspace.materialize_missing_states[index] =
      pack_terminal_missing_state(first_missing_depth, filled);
  workspace.materialize_parent_indices[index] = existing_parent_index;

  const std::uint32_t start_depth =
      request_start_depth(requests, index, leaf_key, first_missing_depth);

  workspace.materialize_node_offsets[index] =
      terminal_node_emit_count(start_depth, target_depth);
  workspace.materialize_leaf_offsets[index] =
      terminal_leaf_emit_count(request, start_depth);
}

__global__ void initialize_storage_kernel(DeviceGpuSvo svo,
                                          const TerminalRequest *requests,
                                          std::uint32_t request_count,
                                          TerminalCompactAllocationWorkspace workspace) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= request_count)
    return;

  const std::uint32_t filled = terminal_missing_state_inherited_filled(
      workspace.materialize_missing_states[index]);
  GpuSvoNode new_node{};
  for (std::uint32_t child = 0u; child < kGroupSize; ++child)
    new_node.child_data[child] = make_uniform_child_data(filled != 0u);

  const std::uint32_t target_depth = workspace.materialize_target_depths[index];
  for (std::uint32_t depth = 1u; depth <= target_depth && depth < kMaxDepth;
       ++depth) {
    if (!node_slot_emits(requests, workspace, index, depth)) {
      continue;
    }
    const std::uint32_t rank =
        emitted_node_rank(requests, workspace, index, depth);
    const std::uint32_t node_index =
        terminal_materialized_node_index(svo, workspace.allocation_state, rank);
    if (node_index < svo.max_node_count)
      svo.nodes[node_index] = new_node;
  }

  if (leaf_slot_emits(requests, workspace, index)) {
    const std::uint32_t rank = leaf_base(workspace, index);
    const std::uint32_t leaf_index =
        terminal_materialized_leaf_index(svo, workspace.allocation_state, rank);
    if (leaf_index < svo.max_leaf_count) {
      svo.leaves[leaf_index] = filled != 0u
                                   ? GpuSvoLeaf{0xffffffffu, 0xffffffffu}
                                   : GpuSvoLeaf{0u, 0u};
    }
  }
}

__global__ void link_storage_kernel(DeviceGpuSvo svo,
                                    const TerminalRequest *requests,
                                    std::uint32_t request_count,
                                    TerminalCompactAllocationWorkspace workspace) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= request_count)
    return;

  const std::uint32_t target_depth = workspace.materialize_target_depths[index];
  const std::uint32_t first_missing_depth =
      terminal_missing_state_first_missing_depth(
          workspace.materialize_missing_states[index]);
  if (first_missing_depth == 0u)
    return;

  const bool filled = terminal_missing_state_inherited_filled(
                          workspace.materialize_missing_states[index]) != 0u;
  const std::uint32_t leaf_key =
      terminal_request_representative_leaf_key(requests[index]);
  const std::uint32_t start_depth =
      recovered_start_depth(requests, workspace, index);
  if (start_depth == 0u)
    return;

  for (std::uint32_t depth = start_depth; depth <= target_depth; ++depth) {
    bool emits = false;
    std::uint32_t child_storage_index = 0u;
    if (depth < kMaxDepth) {
      emits = node_slot_emits(requests, workspace, index, depth);
      if (emits) {
        child_storage_index = terminal_materialized_node_index(
            svo, workspace.allocation_state,
            emitted_node_rank(requests, workspace, index, depth));
      }
    } else {
      emits = leaf_slot_emits(requests, workspace, index);
      if (emits) {
        child_storage_index = terminal_materialized_leaf_index(
            svo, workspace.allocation_state, leaf_base(workspace, index));
      }
    }
    if (!emits)
      continue;

    std::uint32_t parent_index = workspace.materialize_parent_indices[index];
    if (depth > start_depth) {
      const std::uint32_t parent_depth = depth - 1u;
      parent_index = terminal_materialized_node_index(
          svo, workspace.allocation_state,
          emitted_node_rank(requests, workspace, index, parent_depth));
    } else if (start_depth > first_missing_depth) {
      const std::uint32_t parent_depth = depth - 1u;
      const std::uint32_t owner = find_parent_owner(
          requests, workspace, request_count, leaf_key, parent_depth);
      if (owner >= request_count)
        continue;
      parent_index = terminal_materialized_node_index(
          svo, workspace.allocation_state,
          emitted_node_rank(requests, workspace, owner, parent_depth));
    }
    if (parent_index >= svo.max_node_count)
      continue;

    const std::uint32_t child_index =
        child_index_for_leaf_key(leaf_key, depth - 1u);
    svo.nodes[parent_index].child_data[child_index] =
        make_child_data(child_storage_index, filled);
  }
}

__global__ void commit_state_kernel(DeviceGpuSvo svo,
                                    const std::uint32_t *node_offsets,
                                    const std::uint32_t *leaf_offsets,
                                    std::uint32_t request_count,
                                    const AllocationState *state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || request_count == 0u)
    return;

  const std::uint32_t node_count = node_offsets[request_count - 1u];
  const std::uint32_t leaf_count = leaf_offsets[request_count - 1u];
  const std::uint32_t reused_nodes = node_count < state->base_free_node_count
                                         ? node_count
                                         : state->base_free_node_count;
  const std::uint32_t reused_leaves = leaf_count < state->base_free_leaf_count
                                          ? leaf_count
                                          : state->base_free_leaf_count;
  svo.counters->node_count =
      state->base_node_count + (node_count - reused_nodes);
  svo.counters->free_node_count = state->base_free_node_count - reused_nodes;
  svo.counters->leaf_count =
      state->base_leaf_count + (leaf_count - reused_leaves);
  svo.counters->free_leaf_count = state->base_free_leaf_count - reused_leaves;
}

inline cudaError_t allocate_paths(DeviceGpuSvo svo,
                                  const TerminalRequest *requests,
                                  std::uint32_t request_count,
                                  TerminalCompactAllocationWorkspace &workspace,
                                  cudaStream_t stream) {
  if (request_count == 0u)
    return cudaSuccess;

  const std::uint32_t blocks = block_count(request_count, kKernelBlockSize);
  collect_counts_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      svo, requests, request_count, workspace);
  cudaError_t status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  status = algo::cuda::scan::inclusive_sum(
      workspace.materialize_node_offsets, request_count,
      workspace.scan_workspace, workspace.scan_workspace_size, stream);
  if (status != cudaSuccess)
    return status;

  status = algo::cuda::scan::inclusive_sum(
      workspace.materialize_leaf_offsets, request_count,
      workspace.scan_workspace, workspace.scan_workspace_size, stream);
  if (status != cudaSuccess)
    return status;

  snapshot_allocation_state_kernel<<<1, 1, 0, stream>>>(
      svo, workspace.allocation_state);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  initialize_storage_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      svo, requests, request_count, workspace);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  link_storage_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      svo, requests, request_count, workspace);
  status = last_launch_status();
  if (status != cudaSuccess)
    return status;

  commit_state_kernel<<<1, 1, 0, stream>>>(
      svo, workspace.materialize_node_offsets,
      workspace.materialize_leaf_offsets, request_count,
      workspace.allocation_state);
  return last_launch_status();
}

} // namespace allocation::terminal_compact_all_depth

inline cudaError_t
terminal_allocation_impl<TerminalCompactAllDepthAllocation>::allocate_paths(
    DeviceGpuSvo svo, const TerminalRequest *requests,
    std::uint32_t request_count, Workspace &workspace, cudaStream_t stream) {
  return allocation::terminal_compact_all_depth::allocate_paths(
      svo, requests, request_count, workspace, stream);
}

} // namespace algo::svt::cuda::detail
