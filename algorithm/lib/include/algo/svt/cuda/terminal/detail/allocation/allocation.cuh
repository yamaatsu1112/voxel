#pragma once

#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/common.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/compact_all_depth/pipeline.cuh>
#include <algo/svt/cuda/terminal/detail/allocation/plain_depthwise/pipeline.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>
#include <algo/svt/cuda/terminal/edit_config.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

template <class Allocation>
inline cudaError_t allocate_terminal_request_paths(
    DeviceGpuSvo svo, const TerminalRequest *requests,
    std::uint32_t request_count, std::uint32_t *request_keys,
    void *phase_scratch,
    cudaStream_t stream) {
  typename terminal_allocation_impl<Allocation>::Workspace workspace =
      terminal_allocation_impl<Allocation>::create_phase_workspace(
          phase_scratch, request_keys, request_count);
  return terminal_allocation_impl<Allocation>::allocate_paths(
      svo, requests, request_count, workspace, stream);
}

} // namespace algo::svt::cuda::detail
