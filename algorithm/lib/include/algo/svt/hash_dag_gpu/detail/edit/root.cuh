#pragma once

#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/workspace/edit_batch.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

__global__ void update_root_ref_kernel(DeviceHashDagGpu dag,
                                       const std::uint32_t *request_count,
                                       const std::uint32_t *canonical_refs,
                                       const std::uint32_t *edit_status) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  // Publish the new root only after every device phase has succeeded. On error,
  // the old root remains visible and partially inserted canonical nodes are
  // harmless because the DAG tables are content-addressed.
  if (*request_count == 0u || *edit_status != kEditStatusOk)
    return;
  *dag.root_ref = canonical_refs[0];
}

inline cudaError_t publish_root_ref(DeviceHashDagGpu dag,
                                    EditWorkspace &workspace,
                                    cudaStream_t stream) {
  update_root_ref_kernel<<<1, 1, 0, stream>>>(dag, workspace.request_count,
                                              workspace.canonical_refs,
                                              workspace.edit_status);
  return last_launch_status();
}

inline cudaError_t read_edit_status(const std::uint32_t *device_status,
                                    cudaStream_t stream) {
  std::uint32_t host_status = kEditStatusOk;
  cudaError_t status =
      cudaMemcpyAsync(&host_status, device_status, sizeof(host_status),
                      cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess)
    return status;
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess)
    return status;
  return host_status == kEditStatusOk ? cudaSuccess : cudaErrorMemoryAllocation;
}

} // namespace algo::svt::hash_dag_gpu::detail
