#pragma once

namespace algo::cuda::scan::detail {

template <class Op, class T>
__global__ void init_shifted_exclusive_kernel(T* dst, const T* src,
                                              std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    dst[index] = index == 0 ? Op::identity() : src[index - 1];
}

template <class Op, class T>
__global__ void hillis_steele_step_kernel(T* dst, const T* src,
                                          std::uint32_t count,
                                          std::uint32_t offset) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    T value = src[index];
    if (index >= offset) value = Op{}(src[index - offset], value);
    dst[index] = value;
}

template <class Op, class T, int BlockSize>
inline cudaError_t init_shifted_exclusive(T* dst, const T* src,
                                          std::uint32_t count,
                                          cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    init_shifted_exclusive_kernel<Op><<<grid, BlockSize, 0, stream>>>(
        dst, src, count);
    return cudaGetLastError();
}

template <class Op, class T, int BlockSize>
inline cudaError_t hillis_steele_step(T* dst, const T* src,
                                      std::uint32_t count,
                                      std::uint32_t offset,
                                      cudaStream_t stream) {
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    hillis_steele_step_kernel<Op><<<grid, BlockSize, 0, stream>>>(
        dst, src, count, offset);
    return cudaGetLastError();
}

} // namespace algo::cuda::scan::detail
