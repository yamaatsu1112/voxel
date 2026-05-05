#pragma once

#include <algo/cuda/scan/block_based/layout.cuh>

namespace algo::cuda::scan::detail {

template <class Layout, class Op, class T, int BlockSize, int ItemsPerThread>
__device__ __forceinline__ T blelloch_block_exclusive_scan_impl(T* shared) {
    constexpr std::uint32_t kTileSize = BlockSize * ItemsPerThread;
    using layout_impl = shared_layout_impl<Layout, T>;
    const std::uint32_t tid = threadIdx.x;
    const Op op{};

    for (std::uint32_t stride = 1; stride < kTileSize; stride <<= 1) {
        const std::uint32_t work_count = kTileSize / (stride << 1);
        for (std::uint32_t work = tid; work < work_count; work += BlockSize) {
            const std::uint32_t index = (work + 1) * (stride << 1) - 1;
            shared[layout_impl::map(index)] = op(
                shared[layout_impl::map(index - stride)],
                shared[layout_impl::map(index)]);
        }
        __syncthreads();
    }

    T total = Op::identity();
    if (tid == 0) {
        total = shared[layout_impl::map(kTileSize - 1)];
        shared[layout_impl::map(kTileSize - 1)] = Op::identity();
    }
    __syncthreads();

    for (std::uint32_t stride = kTileSize >> 1; stride >= 1; stride >>= 1) {
        const std::uint32_t work_count = kTileSize / (stride << 1);
        for (std::uint32_t work = tid; work < work_count; work += BlockSize) {
            const std::uint32_t index = (work + 1) * (stride << 1) - 1;
            const std::uint32_t left = index - stride;
            const std::uint32_t left_physical = layout_impl::map(left);
            const std::uint32_t index_physical = layout_impl::map(index);
            const T temp = shared[left_physical];
            shared[left_physical] = shared[index_physical];
            shared[index_physical] = op(shared[index_physical], temp);
        }
        __syncthreads();
        if (stride == 1) break;
    }

    return total;
}

} // namespace algo::cuda::scan::detail
