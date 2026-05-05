#pragma once

#include <cstdint>

namespace algo::cuda::scan::detail {

template <class Op, class T, int BlockSize, int ItemsPerThread>
__device__ __forceinline__ T hillis_steele_block_inclusive_scan(T* shared) {
    constexpr std::uint32_t kTileSize = BlockSize * ItemsPerThread;
    const Op op{};

    for (std::uint32_t stride = 1; stride < kTileSize; stride <<= 1) {
        T addends[ItemsPerThread];
        for (int item = 0; item < ItemsPerThread; ++item) {
            const std::uint32_t local_index = threadIdx.x + item * BlockSize;
            addends[item] =
                local_index >= stride ? shared[local_index - stride]
                                      : Op::identity();
        }
        __syncthreads();

        for (int item = 0; item < ItemsPerThread; ++item) {
            const std::uint32_t local_index = threadIdx.x + item * BlockSize;
            if (local_index >= stride)
                shared[local_index] = op(addends[item], shared[local_index]);
        }
        __syncthreads();
    }

    return shared[kTileSize - 1];
}

} // namespace algo::cuda::scan::detail
