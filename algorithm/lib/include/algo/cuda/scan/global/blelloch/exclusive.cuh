#pragma once

namespace algo::cuda::scan::detail {

template <class Op, class T>
__global__ void blelloch_exclusive_upsweep_kernel(
    T* data, std::uint32_t padded_count, std::uint32_t stride) {
    const std::uint32_t work_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t elements_per_group = stride << 1;
    const std::uint32_t right = (work_index + 1) * elements_per_group - 1;
    if (right >= padded_count) return;
    data[right] = Op{}(data[right - stride], data[right]);
}

template <class Op, class T>
__global__ void blelloch_exclusive_set_last_zero_kernel(
    T* data, std::uint32_t padded_count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    data[padded_count - 1] = Op::identity();
}

template <class Op, class T>
__global__ void blelloch_exclusive_downsweep_kernel(
    T* data, std::uint32_t padded_count, std::uint32_t stride) {
    const std::uint32_t work_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t elements_per_group = stride << 1;
    const std::uint32_t right = (work_index + 1) * elements_per_group - 1;
    if (right >= padded_count) return;
    const std::uint32_t left = right - stride;
    const T temp = data[left];
    data[left] = data[right];
    data[right] = Op{}(data[right], temp);
}

template <class Op, class T, int BlockSize>
inline cudaError_t blelloch_exclusive(T* data, std::uint32_t padded_count,
                                      cudaStream_t stream) {
    for (std::uint32_t stride = 1; stride < padded_count; stride <<= 1) {
        const std::uint32_t work_count = padded_count / (stride << 1);
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(work_count, BlockSize));
        blelloch_exclusive_upsweep_kernel<Op>
            <<<grid, BlockSize, 0, stream>>>(
            data, padded_count, stride);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }

    blelloch_exclusive_set_last_zero_kernel<Op><<<1, 1, 0, stream>>>(
        data, padded_count);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    for (std::uint32_t stride = padded_count >> 1; stride >= 1; stride >>= 1) {
        const std::uint32_t work_count = padded_count / (stride << 1);
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(work_count, BlockSize));
        blelloch_exclusive_downsweep_kernel<Op>
            <<<grid, BlockSize, 0, stream>>>(
            data, padded_count, stride);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        if (stride == 1) break;
    }
    return cudaSuccess;
}

} // namespace algo::cuda::scan::detail
