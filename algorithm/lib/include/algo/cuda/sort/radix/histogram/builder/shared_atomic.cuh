#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/sort/radix/common.cuh>

namespace algo::cuda::sort::detail {

template <int BlockSize, int ItemsPerThread, int RadixBits, class Key,
          class OutputLayout>
__global__ void
compute_block_histograms_shared_atomic_kernel(std::uint32_t* histograms,
                                              const Key* keys,
                                              std::uint32_t count, int shift) {
    constexpr std::uint32_t kNumBuckets =
        static_cast<std::uint32_t>(1u << RadixBits);
    __shared__ std::uint32_t histogram[kNumBuckets];

    for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
         bucket += blockDim.x) {
        histogram[bucket] = 0;
    }
    __syncthreads();

    const std::uint32_t tile_base =
        blockIdx.x * static_cast<std::uint32_t>(BlockSize * ItemsPerThread);
    for (int item = 0; item < ItemsPerThread; ++item) {
        const std::uint32_t index =
            warp_major_tile_index<BlockSize, ItemsPerThread>(tile_base, item);
        if (index < count) {
            const std::uint32_t digit =
                extract_digit<RadixBits>(keys[index], shift);
            atomicAdd(&histogram[digit], 1u);
        }
    }
    __syncthreads();

    const std::uint32_t num_blocks = gridDim.x;
    for (std::uint32_t bucket = threadIdx.x; bucket < kNumBuckets;
         bucket += blockDim.x) {
        histograms[OutputLayout::block_histogram_index(bucket, blockIdx.x,
                                                       num_blocks)] =
            histogram[bucket];
    }
}

template <class Key, int BlockSize, int RadixBits, int ItemsPerThread,
          class OutputLayout>
cudaError_t
compute_block_histograms_shared_atomic(std::uint32_t* histograms,
                                       const Key* keys, std::uint32_t count,
                                       int shift, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto grid = static_cast<unsigned int>(::algo::ceil_div(
        count, static_cast<std::uint32_t>(BlockSize * ItemsPerThread)));
    compute_block_histograms_shared_atomic_kernel<BlockSize, ItemsPerThread,
                                                  RadixBits, Key, OutputLayout>
        <<<grid, BlockSize, 0, stream>>>(histograms, keys, count, shift);
    return cudaGetLastError();
}

template <> struct histogram_build_impl<SharedAtomicHistogram> {
    template <class Key, int BlockSize, int RadixBits, int ItemsPerThread,
              class OutputLayout>
    static cudaError_t run(std::uint32_t* histograms, const Key* input,
                           std::uint32_t count, int shift,
                           cudaStream_t stream) {
        return compute_block_histograms_shared_atomic<Key, BlockSize,
                                                      RadixBits,
                                                      ItemsPerThread,
                                                      OutputLayout>(
            histograms, input, count, shift, stream);
    }
};

} // namespace algo::cuda::sort::detail
