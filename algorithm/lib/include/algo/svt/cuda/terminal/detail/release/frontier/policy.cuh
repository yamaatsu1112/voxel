#pragma once

#include <algo/svt/cuda/terminal/detail/release/common.cuh>
#include <algo/svt/cuda/terminal/detail/release/frontier/workspace.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

template <> struct terminal_release_impl<TerminalFrontierRelease> {
  using Workspace = TerminalFrontierReleaseWorkspace;

  static std::size_t release_workspace_size(std::uint32_t request_capacity) {
    return terminal_frontier_release_workspace_size(request_capacity);
  }

  static Workspace create_phase_workspace(
      void *workspace, TerminalDetachedChild *detached_children,
      std::uint32_t *detached_child_count, std::uint32_t request_capacity) {
    return create_terminal_frontier_release_workspace(
        workspace, detached_children, detached_child_count, request_capacity);
  }

  static cudaError_t release(DeviceGpuSvo svo, cudaError_t *status,
                             Workspace workspace, cudaStream_t stream);
};

} // namespace algo::svt::cuda::detail
