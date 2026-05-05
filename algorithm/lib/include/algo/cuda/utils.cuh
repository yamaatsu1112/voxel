#pragma once

#include <algo/utils.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace algo::cuda {

template <class T>
T* device_alloc(std::uint32_t count) {
    if (count == 0) return nullptr;
    T* ptr = nullptr;
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * count);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                 cudaGetErrorString(status));
    }
    return ptr;
}

template <class T>
void device_free(T* ptr) {
    if (ptr == nullptr) return;
    const cudaError_t status = cudaFree(ptr);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaFree failed: ") +
                                 cudaGetErrorString(status));
    }
}

template <class T>
void copy_to_device(T* dst, const T* src, std::uint32_t count) {
    if (count == 0) return;
    const cudaError_t status =
        cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy host to device failed: ") +
            cudaGetErrorString(status));
    }
}

template <class T>
void copy_to_host(T* dst, const T* src, std::uint32_t count) {
    if (count == 0) return;
    const cudaError_t status =
        cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy device to host failed: ") +
            cudaGetErrorString(status));
    }
}

inline void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " +
                                 cudaGetErrorString(status));
    }
}

template <class T>
__global__ void copy_kernel(T* dst, const T* src, std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    dst[index] = src[index];
}

template <class T, int BlockSize>
inline cudaError_t copy_buffer(T* dst, const T* src, std::uint32_t count,
                               cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto grid = static_cast<unsigned int>(algo::ceil_div(count, BlockSize));
    copy_kernel<<<grid, BlockSize, 0, stream>>>(dst, src, count);
    return cudaGetLastError();
}

} // namespace algo::cuda
