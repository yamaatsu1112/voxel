#pragma once

namespace algo::cuda::scan::detail {

struct blelloch_dedicated_inclusive {};
struct blelloch_from_exclusive_inclusive {};

template <class Op, class T>
__global__ void fill_padded_kernel(T* dst, const T* src, std::uint32_t count,
                                   std::uint32_t padded_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= padded_count) return;
    dst[index] = index < count ? src[index] : Op::identity();
}

template <class Op, class T, int BlockSize>
inline cudaError_t fill_padded(T* dst, const T* src, std::uint32_t count,
                               std::uint32_t padded_count,
                               cudaStream_t stream) {
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(padded_count, BlockSize));
    fill_padded_kernel<Op><<<grid, BlockSize, 0, stream>>>(dst, src, count,
                                                           padded_count);
    return cudaGetLastError();
}

} // namespace algo::cuda::scan::detail
