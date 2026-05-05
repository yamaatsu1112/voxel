#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace cuda_test {

inline bool has_cuda_device(cudaError_t* status_out = nullptr) {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status_out)
        *status_out = status;
    return status == cudaSuccess && count > 0;
}

template <class T> T* managed_alloc(std::size_t count) {
    if (count == 0)
        return nullptr;

    T* ptr = nullptr;
    const cudaError_t status =
        cudaMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * count);
    if (status != cudaSuccess) {
        ADD_FAILURE() << "cudaMallocManaged failed: "
                      << cudaGetErrorString(status);
        return nullptr;
    }
    return ptr;
}

template <class T> T* device_alloc(std::size_t count) {
    if (count == 0)
        return nullptr;

    T* ptr = nullptr;
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * count);
    if (status != cudaSuccess) {
        ADD_FAILURE() << "cudaMalloc failed: " << cudaGetErrorString(status);
        return nullptr;
    }
    return ptr;
}

inline void sync_cuda() { ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess); }

template <class T> void free_managed(T* ptr) {
    if (ptr)
        ASSERT_EQ(cudaFree(ptr), cudaSuccess);
}

template <class T> void fill_device(T* dst, const std::vector<T>& values) {
    for (std::size_t i = 0; i < values.size(); ++i)
        dst[i] = values[i];
}

template <class T> std::vector<T> read_device(const T* src, std::size_t count) {
    std::vector<T> out(count);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = src[i];
    return out;
}

template <class T> void copy_to_device(T* dst, const std::vector<T>& values) {
    if (values.empty())
        return;

    ASSERT_EQ(cudaMemcpy(dst, values.data(), sizeof(T) * values.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
}

template <class T> std::vector<T> copy_from_device(const T* src,
                                                   std::size_t count) {
    std::vector<T> values(count);
    if (count == 0)
        return values;

    const cudaError_t status =
        cudaMemcpy(values.data(), src, sizeof(T) * count,
                   cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        ADD_FAILURE() << "cudaMemcpy device-to-host failed: "
                      << cudaGetErrorString(status);
    }
    return values;
}

template <class T> T copy_scalar_from_device(const T* src) {
    T value{};
    const cudaError_t status =
        cudaMemcpy(&value, src, sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        ADD_FAILURE() << "cudaMemcpy scalar device-to-host failed: "
                      << cudaGetErrorString(status);
    }
    return value;
}

} // namespace cuda_test
