#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/common.cuh>

namespace algo::cuda::sort::detail {

template <class> inline constexpr bool kAlwaysFalseOneSweepGlobalOffsets = false;

template <class GlobalOffsetsBlockHistogramPolicy>
struct onesweep_global_offsets_block_histogram_impl {
  static_assert(
      kAlwaysFalseOneSweepGlobalOffsets<GlobalOffsetsBlockHistogramPolicy>,
      "onesweep global offsets block histogram policy is not implemented");
};

template <>
struct onesweep_global_offsets_block_histogram_impl<
    SharedAtomicGlobalOffsetsBlockHistogram> {
  template <int BlockSize, int RadixBits, int KeyBits, class Key>
  __device__ static void compute(std::uint32_t *histograms, const Key *keys,
                                 std::uint32_t count) {
    constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);
    constexpr std::uint32_t kPassCount =
        static_cast<std::uint32_t>(KeyBits / RadixBits);

    for (std::uint32_t index = threadIdx.x; index < kPassCount * kNumBuckets;
         index += blockDim.x) {
      histograms[index] = 0;
    }
    __syncthreads();

    const std::uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index < count) {
      const Key key = keys[global_index];
      for (std::uint32_t pass = 0; pass < kPassCount; ++pass) {
        const int shift =
            static_cast<int>(pass * static_cast<std::uint32_t>(RadixBits));
        const std::uint32_t digit = extract_digit<RadixBits>(key, shift);
        atomicAdd(&histograms[pass * kNumBuckets + digit], 1u);
      }
    }
    __syncthreads();
  }
};

template <int BlockSize, int RadixBits, int KeyBits,
          class GlobalOffsetsBlockHistogramPolicy, class Key>
__global__ void
build_onesweep_global_histogram_kernel(std::uint32_t *global_histogram,
                                       const Key *keys, std::uint32_t count) {
  constexpr std::uint32_t kNumBuckets =
      static_cast<std::uint32_t>(1u << RadixBits);
  constexpr std::uint32_t kPassCount =
      static_cast<std::uint32_t>(KeyBits / RadixBits);

  __shared__ std::uint32_t histograms[kPassCount * kNumBuckets];

  onesweep_global_offsets_block_histogram_impl<
      GlobalOffsetsBlockHistogramPolicy>::template compute<BlockSize, RadixBits,
                                                           KeyBits>(histograms,
                                                                    keys,
                                                                    count);

  for (std::uint32_t index = threadIdx.x; index < kPassCount * kNumBuckets;
       index += blockDim.x) {
    atomicAdd(&global_histogram[index], histograms[index]);
  }
}

template <int RadixBits, int KeyBits>
__global__ void
scan_onesweep_global_offsets_kernel(std::uint32_t *global_offsets) {
  constexpr std::uint32_t kNumBuckets =
      static_cast<std::uint32_t>(1u << RadixBits);
  constexpr std::uint32_t kPassCount =
      static_cast<std::uint32_t>(KeyBits / RadixBits);

  const std::uint32_t pass = threadIdx.x;
  if (pass >= kPassCount)
    return;

  std::uint32_t running = 0;
  for (std::uint32_t bucket = 0; bucket < kNumBuckets; ++bucket) {
    const std::size_t index =
        static_cast<std::size_t>(pass) * kNumBuckets + bucket;
    const std::uint32_t count = global_offsets[index];
    global_offsets[index] = running;
    running += count;
  }
}

template <class Key, int BlockSize, int RadixBits, int KeyBits,
          class GlobalOffsetsBlockHistogramPolicy>
cudaError_t build_onesweep_global_offsets(std::uint32_t *global_offsets,
                                          const Key *keys, std::uint32_t count,
                                          cudaStream_t stream,
                                          GlobalOffsetsBlockHistogramPolicy) {
  constexpr std::uint32_t kNumBuckets =
      static_cast<std::uint32_t>(1u << RadixBits);
  constexpr std::uint32_t kPassCount =
      static_cast<std::uint32_t>(KeyBits / RadixBits);

  cudaError_t status = fill_zero<std::uint32_t, BlockSize>(
      global_offsets, kPassCount * kNumBuckets, stream);
  if (status != cudaSuccess)
    return status;

  if (count != 0) {
    const auto grid =
        static_cast<unsigned int>(::algo::ceil_div(count, BlockSize));
    build_onesweep_global_histogram_kernel<
        BlockSize, RadixBits, KeyBits, GlobalOffsetsBlockHistogramPolicy>
        <<<grid, BlockSize, 0, stream>>>(global_offsets, keys, count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return status;
  }

  scan_onesweep_global_offsets_kernel<RadixBits, KeyBits>
      <<<1, kPassCount, 0, stream>>>(global_offsets);
  return cudaGetLastError();
}

} // namespace algo::cuda::sort::detail
