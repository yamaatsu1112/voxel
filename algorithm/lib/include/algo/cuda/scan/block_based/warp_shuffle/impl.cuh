#pragma once

namespace algo::cuda::scan::detail {

template <int BlockSize, int ItemsPerThread>
struct block_scan_impl<WarpShuffleBlock<BlockSize, ItemsPerThread>> {
    static_assert(BlockSize > 0, "WarpShuffleBlock requires BlockSize > 0");
    static_assert(BlockSize % 32 == 0,
                  "WarpShuffleBlock requires BlockSize to be a multiple of 32");
    static_assert(BlockSize <= 1024,
                  "WarpShuffleBlock requires BlockSize <= 1024");
    static_assert(ItemsPerThread > 0,
                  "WarpShuffleBlock requires ItemsPerThread > 0");
    static constexpr std::uint32_t kTileSize = BlockSize * ItemsPerThread;

    template <class Op, class T>
    static cudaError_t inclusive_scan(T* d_data, std::uint32_t count,
                                      T* block_sums, cudaStream_t stream) {
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(count, kTileSize));
        block_scan_kernel<true, WarpShuffleBlock<BlockSize, ItemsPerThread>,
                          Op, T>
            <<<grid, BlockSize, 0, stream>>>(d_data, count, block_sums);
        return cudaGetLastError();
    }

    template <class Op, class T>
    static cudaError_t exclusive_scan(T* d_data, std::uint32_t count,
                                      T* block_sums, cudaStream_t stream) {
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(count, kTileSize));
        block_scan_kernel<false, WarpShuffleBlock<BlockSize, ItemsPerThread>,
                          Op, T>
            <<<grid, BlockSize, 0, stream>>>(d_data, count, block_sums);
        return cudaGetLastError();
    }

    template <class Op, class T>
    static cudaError_t add_block_offsets(T* d_data, std::uint32_t count,
                                         const T* block_offsets,
                                         cudaStream_t stream) {
        return block_scan_add_block_offsets<
            WarpShuffleBlock<BlockSize, ItemsPerThread>, Op, T>(
            d_data, count, block_offsets, stream);
    }
};

} // namespace algo::cuda::scan::detail
