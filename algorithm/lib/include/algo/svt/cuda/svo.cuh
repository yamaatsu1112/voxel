#pragma once

#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/detail/reset.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/query.cuh>
#include <algo/svt/cuda/storage.cuh>

#include <cuda_runtime.h>

namespace algo::svt::cuda {

// Initialize counters, root node state, and free lists inside an already
// allocated GpuSvo view.
inline cudaError_t reset_svo(DeviceGpuSvo svo, cudaStream_t stream = nullptr) {
    if (svo.nodes == nullptr || svo.leaves == nullptr || svo.counters == nullptr)
        return cudaErrorInvalidValue;
    detail::reset_svo_kernel<<<1, 1, 0, stream>>>(svo);
    return detail::last_launch_status();
}

} // namespace algo::svt::cuda
