#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

inline constexpr std::uint32_t kKernelBlockSize = 256u;

inline std::uint32_t block_count(std::uint32_t count,
                                 std::uint32_t block_size) {
    return (count + block_size - 1u) / block_size;
}

inline cudaError_t last_launch_status() {
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ? cudaSuccess : status;
}

} // namespace algo::svt::cuda::detail
