#pragma once

#include <algo/svt/cuda/terminal/detail/release/common.cuh>
#include <algo/svt/cuda/terminal/detail/release/depthwise/workspace.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

template <> struct terminal_release_impl<TerminalDepthwiseRelease> {
  using Workspace = TerminalDepthwiseReleaseWorkspace;

  static std::size_t release_workspace_size(std::uint32_t request_capacity) {
    return terminal_depthwise_release_workspace_size(request_capacity);
  }

  static Workspace create_phase_workspace(
      void *workspace, TerminalDetachedChild *detached_children,
      std::uint32_t *detached_child_count, std::uint32_t request_capacity) {
    return create_terminal_depthwise_release_workspace(
        workspace, detached_children, detached_child_count, request_capacity);
  }

  static cudaError_t release(DeviceGpuSvo svo, cudaError_t *status,
                             Workspace workspace, cudaStream_t stream);
};

} // namespace algo::svt::cuda::detail
