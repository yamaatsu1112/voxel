#pragma once

#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/workspace/edit_batch.cuh>
#include <algo/svt/hash_dag_gpu/detail/node.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

__global__ void initialize_edit_nodes_kernel(DeviceHashDagGpu dag,
                                             const EditSvoRequest *requests,
                                             const std::uint32_t *request_count,
                                             std::uint32_t *old_refs,
                                             std::uint32_t *canonical_refs,
                                             std::uint32_t *child_refs) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= *request_count)
    return;

  const EditSvoRequest request = requests[index];
  const std::uint32_t depth = request_depth(request);
  // Walk the current DAG to the request node. This captures unchanged siblings
  // before edited descendants start replacing child_refs bottom-up.
  std::uint32_t ref = *dag.root_ref;
  for (std::uint32_t level = 0u; level < depth; ++level) {
    const std::uint32_t child =
        child_index_from_prefix(request.prefix, depth, level);
    ref = child_ref(dag, ref, child);
  }

  old_refs[index] = ref;
  canonical_refs[index] = ref;
  for (std::uint32_t child = 0u; child < kGroupSize; ++child)
    child_refs[index * kGroupSize + child] = child_ref(dag, ref, child);
}

inline cudaError_t initialize_edit_nodes(DeviceHashDagGpu dag,
                                         std::uint32_t count,
                                         EditWorkspace &workspace,
                                         cudaStream_t stream) {
  // Snapshot the current refs for every requested node and seed child_refs with
  // the old children. Later phases overwrite only the changed children.
  const std::uint32_t max_requests = max_request_count(count);
  const std::uint32_t blocks = block_count(max_requests, kKernelBlockSize);
  initialize_edit_nodes_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
      dag, workspace.requests, workspace.request_count, workspace.old_refs,
      workspace.canonical_refs, workspace.child_refs);
  return last_launch_status();
}

} // namespace algo::svt::hash_dag_gpu::detail
