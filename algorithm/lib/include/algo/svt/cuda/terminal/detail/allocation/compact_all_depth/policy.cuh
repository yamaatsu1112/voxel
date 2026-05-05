#pragma once

#include <algo/svt/cuda/terminal/detail/allocation/common.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/compact_all_depth/workspace.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

template <> struct terminal_allocation_impl<TerminalCompactAllDepthAllocation> {
  using Workspace = TerminalCompactAllocationWorkspace;

  static std::size_t allocation_workspace_size(
      std::uint32_t request_capacity) {
    return terminal_compact_allocation_workspace_size(request_capacity);
  }

  static Workspace create_phase_workspace(void *workspace,
                                          std::uint32_t *request_keys,
                                          std::uint32_t request_capacity) {
    (void)request_keys;
    return create_terminal_compact_allocation_workspace(workspace,
                                                        request_capacity);
  }

  static cudaError_t allocate_paths(DeviceGpuSvo svo,
                                    const TerminalRequest *requests,
                                    std::uint32_t request_count,
                                    Workspace &workspace, cudaStream_t stream);
};

} // namespace algo::svt::cuda::detail
