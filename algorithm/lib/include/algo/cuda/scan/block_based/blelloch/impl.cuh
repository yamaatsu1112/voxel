#pragma once

namespace algo::cuda::scan::detail {

template <int BlockSize, int ItemsPerThread, class Layout>
struct block_scan_impl<BlellochBlock<BlockSize, ItemsPerThread, Layout>> {
    static_assert(::algo::is_power_of_two(BlockSize),
                  "Blelloch shared scan requires power-of-two BlockSize");
    static_assert(::algo::is_power_of_two(ItemsPerThread),
                  "Blelloch shared scan requires power-of-two ItemsPerThread");
    static constexpr std::uint32_t kTileSize = BlockSize * ItemsPerThread;

    template <class Op, class T>
    static cudaError_t inclusive_scan(T* d_data, std::uint32_t count,
                                      T* block_sums, cudaStream_t stream) {
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(count, kTileSize));
        block_scan_kernel<true, BlellochBlock<BlockSize, ItemsPerThread, Layout>,
                          Op, T>
            <<<grid, BlockSize, 0, stream>>>(d_data, count, block_sums);
        return cudaGetLastError();
    }

    template <class Op, class T>
    static cudaError_t exclusive_scan(T* d_data, std::uint32_t count,
                                      T* block_sums, cudaStream_t stream) {
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(count, kTileSize));
        block_scan_kernel<false, BlellochBlock<BlockSize, ItemsPerThread, Layout>,
                          Op, T>
            <<<grid, BlockSize, 0, stream>>>(d_data, count, block_sums);
        return cudaGetLastError();
    }

    template <class Op, class T>
    static cudaError_t add_block_offsets(T* d_data, std::uint32_t count,
                                         const T* block_offsets,
                                         cudaStream_t stream) {
        return block_scan_add_block_offsets<
            BlellochBlock<BlockSize, ItemsPerThread, Layout>, Op, T>(
            d_data, count, block_offsets, stream);
    }
};

} // namespace algo::cuda::scan::detail
