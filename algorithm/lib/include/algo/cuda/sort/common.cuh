#pragma once

#include <algo/cuda/utils.cuh>

#include <cstddef>
#include <cstdint>

namespace algo::cuda::sort::detail {

template <class T>
constexpr std::size_t align_up(std::size_t offset) {
    constexpr std::size_t kAlign = alignof(T);
    return (offset + kAlign - 1) & ~(kAlign - 1);
}

template <class T>
T* pointer_at(void* base, std::size_t offset) {
    return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <class T>
__global__ void fill_zero_kernel(T* data, std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    data[index] = T{};
}

template <class T, int BlockSize>
cudaError_t fill_zero(T* data, std::uint32_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    fill_zero_kernel<<<grid, BlockSize, 0, stream>>>(data, count);
    return cudaGetLastError();
}

} // namespace algo::cuda::sort::detail
