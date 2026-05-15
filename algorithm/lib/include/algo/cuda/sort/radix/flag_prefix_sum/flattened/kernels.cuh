#pragma once

#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/value_arrays.cuh>

namespace algo::cuda::sort::detail {

template <int RadixBits, class Key>
__global__ void
scatter_keys_flattened_kernel(Key* output, const Key* input,
                              const std::uint32_t* scanned_flags,
                              std::uint32_t count, int shift) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const std::uint32_t digit = extract_digit<RadixBits>(input[index], shift);
    const std::uint32_t output_index =
        scanned_flags[static_cast<std::size_t>(digit) * count + index];
    output[output_index] = input[index];
}

template <class Key, int BlockSize, int RadixBits>
cudaError_t scatter_keys_flattened(Key* output, const Key* input,
                                   const std::uint32_t* scanned_flags,
                                   std::uint32_t count, int shift,
                                   cudaStream_t stream) {
    if (count == 0)
        return cudaSuccess;
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    scatter_keys_flattened_kernel<RadixBits><<<grid, BlockSize, 0, stream>>>(
        output, input, scanned_flags, count, shift);
    return cudaGetLastError();
}

template <int RadixBits, class Key, class... Values>
__global__ void
scatter_by_key_flattened_kernel(Key* output_keys,
                                value_arrays_t<Values...> output_values,
                                const Key* input_keys,
                                value_arrays_t<Values...> input_values,
                                const std::uint32_t* scanned_flags,
                                std::uint32_t count, int shift) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const std::uint32_t digit =
        extract_digit<RadixBits>(input_keys[index], shift);
    const std::uint32_t output_index =
        scanned_flags[static_cast<std::size_t>(digit) * count + index];
    output_keys[output_index] = input_keys[index];
    copy_value_array_item(output_values, output_index, input_values, index);
}

template <class Key, int BlockSize, int RadixBits, class... Values>
cudaError_t
scatter_by_key_flattened(Key* output_keys,
                         value_arrays_t<Values...> output_values,
                         const Key* input_keys,
                         value_arrays_t<Values...> input_values,
                         const std::uint32_t* scanned_flags,
                         std::uint32_t count, int shift,
                         cudaStream_t stream) {
    if (count == 0)
        return cudaSuccess;
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    scatter_by_key_flattened_kernel<RadixBits>
        <<<grid, BlockSize, 0, stream>>>(output_keys, output_values,
                                         input_keys, input_values,
                                         scanned_flags, count, shift);
    return cudaGetLastError();
}

} // namespace algo::cuda::sort::detail
