#pragma once

#include <algo/svt/hash_dag_gpu/detail/edit_pipeline.cuh>
#include <algo/svt/hash_dag_gpu/edit_config.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::hash_dag_gpu {

// Edits are fully device-side but require caller-owned scratch storage. The
// size depends only on the input count, so callers can cache/reuse one buffer
// for repeated batches.
template <class Config = detail::DefaultEditConfig>
inline std::size_t apply_voxel_edits_workspace_size(std::uint32_t count) {
  return detail::edit_impl<Config>::workspace_size(count);
}

template <class Config = detail::DefaultEditConfig>
inline cudaError_t apply_voxel_edits(DeviceHashDagGpu dag,
                                     const VoxelEdit *edits,
                                     std::uint32_t count, EditOp op,
                                     void *workspace,
                                     std::size_t workspace_size_bytes,
                                     cudaStream_t stream = nullptr) {
  // Validate only the API contract here. Per-coordinate validation is done in
  // the packing kernel so invalid coordinates simply contribute no edit.
  if (dag.root_ref == nullptr || (edits == nullptr && count != 0u))
    return cudaErrorInvalidValue;
  if (count == 0u)
    return cudaSuccess;

  const std::size_t required = apply_voxel_edits_workspace_size<Config>(count);
  if (workspace == nullptr || workspace_size_bytes < required)
    return cudaErrorInvalidValue;

  return detail::edit_impl<Config>::apply(dag, edits, count, op, workspace,
                                          workspace_size_bytes, stream);
}

template <class Config = detail::DefaultEditConfig>
inline cudaError_t place_voxel_edits(DeviceHashDagGpu dag,
                                     const VoxelEdit *edits,
                                     std::uint32_t count, void *workspace,
                                     std::size_t workspace_size_bytes,
                                     cudaStream_t stream = nullptr) {
  return apply_voxel_edits<Config>(dag, edits, count, EditOp::Place, workspace,
                                   workspace_size_bytes, stream);
}

template <class Config = detail::DefaultEditConfig>
inline cudaError_t destroy_voxel_edits(DeviceHashDagGpu dag,
                                       const VoxelEdit *edits,
                                       std::uint32_t count, void *workspace,
                                       std::size_t workspace_size_bytes,
                                       cudaStream_t stream = nullptr) {
  return apply_voxel_edits<Config>(dag, edits, count, EditOp::Destroy,
                                   workspace, workspace_size_bytes, stream);
}

} // namespace algo::svt::hash_dag_gpu
