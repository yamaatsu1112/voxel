#pragma once

#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/detail/edit_common.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/terminal/edit_config.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

template <class Release> struct terminal_release_impl {
  static_assert(
      kAlwaysFalse<Release>,
      "terminal_release_impl is not implemented for this release policy");
};

inline constexpr std::uint32_t kTerminalAllocatingChildData =
    kChildMaskBit | kChildIndexMask;

__device__ inline bool terminal_child_is_allocating(std::uint32_t child_data) {
  return (child_data & (kChildMaskBit | kChildIndexMask)) ==
         kTerminalAllocatingChildData;
}

__device__ inline bool
terminal_release_child_is_active(std::uint32_t child_data) {
  return (child_data & kChildMaskBit) != 0u &&
         !terminal_child_is_allocating(child_data);
}

__device__ inline std::uint32_t terminal_prefix_value(
    const std::uint32_t *counts, std::uint32_t index) {
  return index == 0u ? 0u : counts[index - 1u];
}

__global__ void initialize_terminal_release_frontier_kernel(
    const TerminalDetachedChild *detached_children,
    const std::uint32_t *detached_child_count,
    TerminalDetachedChild *release_frontier,
    std::uint32_t *release_frontier_count, std::uint32_t release_capacity,
    cudaError_t *release_status) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = *detached_child_count;
  if (index == 0u) {
    if (count > release_capacity) {
      *release_status = cudaErrorInvalidValue;
      *release_frontier_count = release_capacity;
    } else {
      *release_frontier_count = count;
    }
  }
  if (index >= count)
    return;
  if (index >= release_capacity) {
    *release_status = cudaErrorInvalidValue;
    return;
  }
  release_frontier[index] = detached_children[index];
}

__global__ void snapshot_terminal_release_state_kernel(DeviceGpuSvo svo,
                                                       AllocationState *state) {
  state->base_node_count = svo.counters->node_count;
  state->base_leaf_count = svo.counters->leaf_count;
  state->base_free_node_count = svo.counters->free_node_count;
  state->base_free_leaf_count = svo.counters->free_leaf_count;
}

__global__ void merge_terminal_release_status_kernel(
    const cudaError_t *release_status, cudaError_t *status) {
  if (*release_status != cudaSuccess)
    *status = *release_status;
}

template <class Release>
inline cudaError_t release_terminal_detached_children(
    DeviceGpuSvo svo, cudaError_t *status,
    typename terminal_release_impl<Release>::Workspace workspace,
    cudaStream_t stream) {
  return terminal_release_impl<Release>::release(svo, status, workspace,
                                                 stream);
}

} // namespace algo::svt::cuda::detail
