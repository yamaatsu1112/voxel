#pragma once

namespace algo::cuda::scan::detail {

template <class InclusiveImpl>
struct blelloch_inclusive_impl;

template <class Op, class T>
__global__ void blelloch_inclusive_upsweep_kernel(T* data,
                                                  std::uint32_t padded_count,
                                                  std::uint32_t stride) {
    const std::uint32_t work_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t elements_per_group = stride << 1;
    const std::uint32_t left = work_index * elements_per_group;
    const std::uint32_t right = left + stride;
    if (right >= padded_count) return;
    data[left] = Op{}(data[left], data[right]);
}

template <class Op, class T>
__global__ void blelloch_inclusive_downsweep_kernel(T* data,
                                                    std::uint32_t padded_count,
                                                    std::uint32_t stride) {
    const std::uint32_t work_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t elements_per_group = stride << 1;
    const std::uint32_t left = work_index * elements_per_group;
    const std::uint32_t right = left + stride;
    if (right >= padded_count) return;
    const T right_sum = data[right];
    data[right] = data[left];
    data[left] = right_sum;
}

template <class Op, class T>
__global__ void add_buffers_kernel(T* dst, const T* lhs, const T* rhs,
                                   std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    dst[index] = Op{}(lhs[index], rhs[index]);
}

template <class Op, class T, int BlockSize>
inline cudaError_t blelloch_inclusive(T* data, std::uint32_t padded_count,
                                      cudaStream_t stream) {
    for (std::uint32_t stride = 1; stride < padded_count; stride <<= 1) {
        const std::uint32_t work_count = padded_count / (stride << 1);
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(work_count, BlockSize));
        blelloch_inclusive_upsweep_kernel<Op>
            <<<grid, BlockSize, 0, stream>>>(
            data, padded_count, stride);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }

    for (std::uint32_t stride = padded_count >> 1; stride >= 1; stride >>= 1) {
        const std::uint32_t work_count = padded_count / (stride << 1);
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(work_count, BlockSize));
        blelloch_inclusive_downsweep_kernel<Op>
            <<<grid, BlockSize, 0, stream>>>(
            data, padded_count, stride);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        if (stride == 1) break;
    }
    return cudaSuccess;
}

template <class Op, class T, int BlockSize>
inline cudaError_t add_buffers(T* dst, const T* lhs, const T* rhs,
                               std::uint32_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    add_buffers_kernel<Op><<<grid, BlockSize, 0, stream>>>(dst, lhs, rhs,
                                                           count);
    return cudaGetLastError();
}

template <>
struct blelloch_inclusive_impl<blelloch_dedicated_inclusive> {
    template <class Op, class T, int BlockSize>
    static cudaError_t run(T* d_data, std::uint32_t count, void* workspace,
                           cudaStream_t stream) {
        if (count <= 1) return cudaSuccess;
        auto* buffer = static_cast<T*>(workspace);
        const std::uint32_t padded_count = ::algo::next_power_of_two(count);
        cudaError_t status = fill_padded<Op, T, BlockSize>(
            buffer, d_data, count, padded_count, stream);
        if (status != cudaSuccess) return status;
        status = blelloch_inclusive<Op, T, BlockSize>(buffer, padded_count,
                                                      stream);
        if (status != cudaSuccess) return status;
        return ::algo::cuda::copy_buffer<T, BlockSize>(d_data, buffer, count,
                                                       stream);
    }
};

template <>
struct blelloch_inclusive_impl<blelloch_from_exclusive_inclusive> {
    template <class Op, class T, int BlockSize>
    static cudaError_t run(T* d_data, std::uint32_t count, void* workspace,
                           cudaStream_t stream) {
        if (count <= 1) return cudaSuccess;
        auto* buffer = static_cast<T*>(workspace);
        const std::uint32_t padded_count = ::algo::next_power_of_two(count);
        cudaError_t status = fill_padded<Op, T, BlockSize>(
            buffer, d_data, count, padded_count, stream);
        if (status != cudaSuccess) return status;
        status = blelloch_exclusive<Op, T, BlockSize>(buffer, padded_count,
                                                      stream);
        if (status != cudaSuccess) return status;
        return add_buffers<Op, T, BlockSize>(d_data, buffer, d_data, count,
                                             stream);
    }
};

} // namespace algo::cuda::scan::detail
