#pragma once

#include <algo/svt/hash_dag_gpu/detail/dispatch_capacity.cuh>
#include <algo/svt/hash_dag_gpu/detail/build_leaf_masks/pipeline.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/leaf_update.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/rebuild.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/requests.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/root.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit/snapshots.cuh>
#include <algo/svt/hash_dag_gpu/detail/workspace/edit_batch.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>
#include <algo/svt/hash_dag_gpu/edit_config.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

using DefaultEditConfig =
    EditConfig<ScanDepthwiseRequests<Fused>, HostLeafCountDispatch>;

template <class> inline constexpr bool kEditAlwaysFalse = false;

template <class Config> struct edit_impl {
  static_assert(kEditAlwaysFalse<Config>,
                "edit_impl is not implemented for this edit config");
};

template <class RequestBuilder, class Dispatch>
inline cudaError_t apply_voxel_edits_impl(DeviceHashDagGpu dag,
                                          const VoxelEdit *edits,
                                          std::uint32_t count, EditOp op,
                                          void *workspace,
                                          cudaStream_t stream) {
  // High-level edit pipeline:
  // 1. Reduce edit coordinates to leaf masks.
  // 2. Materialize the touched leaf/ancestor request DAG.
  // 3. Apply leaf masks.
  // 4. Walk bottom-up canonicalizing parents.
  // 5. Publish the new root ref if no device-side error was reported.
  EditWorkspace typed_workspace = create_edit_workspace(workspace, count);
  cudaError_t status = cudaMemsetAsync(typed_workspace.edit_status, 0,
                                       sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(typed_workspace.leaf_count, 0,
                           sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(typed_workspace.request_count, 0,
                           sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;

  status = build_leaf_masks(edits, count, typed_workspace, stream);
  if (status != cudaSuccess)
    return status;

  std::uint32_t leaf_mask_capacity = 0u;
  status = resolve_dispatch_capacity<Dispatch>(
      typed_workspace.leaf_count, count, &leaf_mask_capacity, stream);
  if (status != cudaSuccess)
    return status;

  status = request_builder_impl<RequestBuilder>::build(
      count, leaf_mask_capacity, typed_workspace, stream);
  if (status != cudaSuccess)
    return status;

  status = initialize_edit_nodes(dag, count, typed_workspace, stream);
  if (status != cudaSuccess)
    return status;

  status = apply_leaf_masks(count, op, dag, typed_workspace, stream);
  if (status != cudaSuccess)
    return status;

  status = scatter_leaves_to_parents(count, typed_workspace, stream);
  if (status != cudaSuccess)
    return status;

  for (std::uint32_t reverse_depth = kInnerDepth; reverse_depth > 0u;
       --reverse_depth) {
    const std::uint32_t depth = reverse_depth - 1u;
    status = process_depth(dag, count, depth, typed_workspace, stream);
    if (status != cudaSuccess)
      return status;
  }

  status = publish_root_ref(dag, typed_workspace, stream);
  if (status != cudaSuccess)
    return status;

  return read_edit_status(typed_workspace.edit_status, stream);
}

template <class RequestBuilder, class Dispatch>
struct edit_impl<EditConfig<RequestBuilder, Dispatch>> {
  static std::size_t workspace_size(std::uint32_t count) {
    return edit_workspace_size(count);
  }

  static cudaError_t apply(DeviceHashDagGpu dag, const VoxelEdit *edits,
                           std::uint32_t count, EditOp op, void *workspace,
                           std::size_t, cudaStream_t stream) {
    return apply_voxel_edits_impl<RequestBuilder, Dispatch>(
        dag, edits, count, op, workspace, stream);
  }
};

} // namespace algo::svt::hash_dag_gpu::detail
