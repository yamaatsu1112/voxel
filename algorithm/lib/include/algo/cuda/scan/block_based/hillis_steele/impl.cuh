#pragma once

namespace algo::cuda::scan::detail {

template <int BlockSize, int ItemsPerThread>
struct block_scan_impl<HillisSteeleBlock<BlockSize, ItemsPerThread>> {
    static_assert(::algo::is_power_of_two(BlockSize),
                  "Hillis-Steele shared scan requires power-of-two BlockSize");
    static_assert(::algo::is_power_of_two(ItemsPerThread),
                  "Hillis-Steele shared scan requires power-of-two ItemsPerThread");
    static constexpr std::uint32_t kTileSize = BlockSize * ItemsPerThread;

    template <class Op, class T>
    static cudaError_t inclusive_scan(T* d_data, std::uint32_t count,
                                      T* block_sums, cudaStream_t stream) {
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(count, kTileSize));
        block_scan_kernel<true, HillisSteeleBlock<BlockSize, ItemsPerThread>,
                          Op, T>
            <<<grid, BlockSize, 0, stream>>>(d_data, count, block_sums);
        return cudaGetLastError();
    }

    template <class Op, class T>
    static cudaError_t exclusive_scan(T* d_data, std::uint32_t count,
                                      T* block_sums, cudaStream_t stream) {
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(count, kTileSize));
        block_scan_kernel<false, HillisSteeleBlock<BlockSize, ItemsPerThread>,
                          Op, T>
            <<<grid, BlockSize, 0, stream>>>(d_data, count, block_sums);
        return cudaGetLastError();
    }

    template <class Op, class T>
    static cudaError_t add_block_offsets(T* d_data, std::uint32_t count,
                                         const T* block_offsets,
                                         cudaStream_t stream) {
        return block_scan_add_block_offsets<
            HillisSteeleBlock<BlockSize, ItemsPerThread>, Op, T>(
            d_data, count, block_offsets, stream);
    }
};

} // namespace algo::cuda::scan::detail
