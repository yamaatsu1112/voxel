#pragma once

#include <algo/svt/cuda/voxel/edit_config.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

template <class> inline constexpr bool kDispatchAlwaysFalse = false;

template <class Dispatch> struct dispatch_capacity_impl {
    static_assert(kDispatchAlwaysFalse<Dispatch>,
                  "dispatch_capacity_impl is not implemented for this policy");
};

template <> struct dispatch_capacity_impl<VoxelCountDispatch> {
    static cudaError_t resolve(const std::uint32_t*, std::uint32_t count,
                               std::uint32_t* capacity, cudaStream_t) {
        *capacity = count;
        return cudaSuccess;
    }
};

template <> struct dispatch_capacity_impl<HostLeafCountDispatch> {
    static cudaError_t resolve(const std::uint32_t* leaf_count,
                               std::uint32_t, std::uint32_t* capacity,
                               cudaStream_t stream) {
        cudaError_t status =
            cudaMemcpyAsync(capacity, leaf_count, sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess)
            return status;
        return cudaStreamSynchronize(stream);
    }
};

template <class Dispatch>
inline cudaError_t resolve_dispatch_capacity(const std::uint32_t* leaf_count,
                                             std::uint32_t count,
                                             std::uint32_t* capacity,
                                             cudaStream_t stream) {
    return dispatch_capacity_impl<Dispatch>::resolve(leaf_count, count,
                                                     capacity, stream);
}

} // namespace algo::svt::cuda::detail
