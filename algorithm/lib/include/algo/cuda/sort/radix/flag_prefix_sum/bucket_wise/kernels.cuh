#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/value_arrays.cuh>

namespace algo::cuda::sort::detail {

template <int RadixBits, class Key>
__global__ void compute_bucket_offsets_kernel(std::uint32_t* bucket_offsets,
                                              const std::uint32_t* flags,
                                              const Key* keys,
                                              std::uint32_t count, int shift) {
    constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);
    const std::uint32_t bucket = threadIdx.x;
    std::uint32_t bucket_total = 0;

    if (bucket < kNumBuckets && count != 0) {
        const std::uint32_t last_index = count - 1;
        const std::uint32_t last_digit =
            extract_digit<RadixBits>(keys[last_index], shift);
        const auto* bucket_flags =
            flags + static_cast<std::size_t>(bucket) * count;
        bucket_total =
            bucket_flags[last_index] + (last_digit == bucket ? 1u : 0u);
    }

    std::uint32_t inclusive_sum = bucket_total;
    constexpr unsigned int kWarpMask = 0xffffffffu;
    for (int delta = 1; delta < 32; delta <<= 1) {
        const std::uint32_t value =
            __shfl_up_sync(kWarpMask, inclusive_sum, delta);
        if (threadIdx.x >= static_cast<unsigned int>(delta)) {
            inclusive_sum += value;
        }
    }

    if (bucket < kNumBuckets) {
        bucket_offsets[bucket] = inclusive_sum - bucket_total;
    }
}

template <class Key, int RadixBits>
cudaError_t compute_bucket_offsets(std::uint32_t* bucket_offsets,
                                   const std::uint32_t* flags, const Key* keys,
                                   std::uint32_t count, int shift,
                                   cudaStream_t stream) {
    compute_bucket_offsets_kernel<RadixBits>
        <<<1, 32, 0, stream>>>(bucket_offsets, flags, keys, count, shift);
    return cudaGetLastError();
}

template <int RadixBits, class Key>
__global__ void scatter_keys_kernel(Key* output, const Key* input,
                                    const std::uint32_t* scanned_flags,
                                    const std::uint32_t* bucket_offsets,
                                    std::uint32_t count, int shift) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const std::uint32_t digit = extract_digit<RadixBits>(input[index], shift);
    const std::uint32_t local_offset =
        scanned_flags[static_cast<std::size_t>(digit) * count + index];
    output[bucket_offsets[digit] + local_offset] = input[index];
}

template <class Key, int BlockSize, int RadixBits>
cudaError_t scatter_keys(Key* output, const Key* input,
                         const std::uint32_t* scanned_flags,
                         const std::uint32_t* bucket_offsets,
                         std::uint32_t count, int shift, cudaStream_t stream) {
    if (count == 0)
        return cudaSuccess;
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    scatter_keys_kernel<RadixBits><<<grid, BlockSize, 0, stream>>>(
        output, input, scanned_flags, bucket_offsets, count, shift);
    return cudaGetLastError();
}

template <int RadixBits, class Key, class... Values>
__global__ void scatter_by_key_kernel(Key* output_keys,
                                      value_arrays_t<Values...> output_values,
                                      const Key* input_keys,
                                      value_arrays_t<Values...> input_values,
                                      const std::uint32_t* scanned_flags,
                                      const std::uint32_t* bucket_offsets,
                                      std::uint32_t count, int shift) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const std::uint32_t digit =
        extract_digit<RadixBits>(input_keys[index], shift);
    const std::uint32_t local_offset =
        scanned_flags[static_cast<std::size_t>(digit) * count + index];
    const std::uint32_t output_index = bucket_offsets[digit] + local_offset;
    output_keys[output_index] = input_keys[index];
    copy_value_array_item(output_values, output_index, input_values, index);
}

template <class Key, int BlockSize, int RadixBits, class... Values>
cudaError_t scatter_by_key(Key* output_keys,
                           value_arrays_t<Values...> output_values,
                           const Key* input_keys,
                           value_arrays_t<Values...> input_values,
                           const std::uint32_t* scanned_flags,
                           const std::uint32_t* bucket_offsets,
                           std::uint32_t count, int shift,
                           cudaStream_t stream) {
    if (count == 0)
        return cudaSuccess;
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    scatter_by_key_kernel<RadixBits><<<grid, BlockSize, 0, stream>>>(
        output_keys, output_values, input_keys, input_values, scanned_flags,
        bucket_offsets, count, shift);
    return cudaGetLastError();
}

} // namespace algo::cuda::sort::detail
