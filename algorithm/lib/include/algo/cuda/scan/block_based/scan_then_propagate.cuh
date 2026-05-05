#pragma once

namespace algo::cuda::scan::detail {

template <class BlockScan>
struct scan_impl<BlockBased<ScanThenPropagate<BlockScan>>> {
    template <class Op, class T>
    static std::size_t required_workspace_size(std::uint32_t count) {
        constexpr std::uint32_t kTileSize = block_scan_impl<BlockScan>::kTileSize;
        if (count <= kTileSize) return 0;
        const std::uint32_t num_blocks = ::algo::ceil_div(count, kTileSize);
        return sizeof(T) * static_cast<std::size_t>(num_blocks) +
               scan_impl<BlockBased<ScanThenPropagate<BlockScan>>>::
                   template required_workspace_size<Op, T>(num_blocks);
    }

    template <class Op, class T>
    static cudaError_t inclusive_scan(T* d_data, std::uint32_t count,
                                      void* workspace,
                                      std::size_t workspace_size,
                                      cudaStream_t stream) {
        return run<true, Op, T>(d_data, count, workspace, workspace_size,
                                stream);
    }

    template <class Op, class T>
    static cudaError_t exclusive_scan(T* d_data, std::uint32_t count,
                                      void* workspace,
                                      std::size_t workspace_size,
                                      cudaStream_t stream) {
        return run<false, Op, T>(d_data, count, workspace, workspace_size,
                                 stream);
    }

private:
    template <bool Inclusive, class Op, class T>
    static cudaError_t run(T* d_data, std::uint32_t count, void* workspace,
                           std::size_t workspace_size, cudaStream_t stream) {
        constexpr std::uint32_t kTileSize = block_scan_impl<BlockScan>::kTileSize;

        if (count == 0) return cudaSuccess;
        if (count <= kTileSize) {
            if constexpr (Inclusive) {
                return block_scan_impl<BlockScan>::
                    template inclusive_scan<Op, T>(d_data, count, nullptr,
                                                   stream);
            } else {
                return block_scan_impl<BlockScan>::
                    template exclusive_scan<Op, T>(d_data, count, nullptr,
                                                   stream);
            }
        }

        const std::uint32_t num_blocks = ::algo::ceil_div(count, kTileSize);
        const std::size_t block_sums_bytes =
            sizeof(T) * static_cast<std::size_t>(num_blocks);
        auto* block_sums = static_cast<T*>(workspace);
        auto* next_workspace = static_cast<void*>(
            static_cast<std::byte*>(workspace) + block_sums_bytes);
        const std::size_t next_workspace_size = workspace_size - block_sums_bytes;

        cudaError_t status = cudaSuccess;
        if constexpr (Inclusive) {
            status = block_scan_impl<BlockScan>::
                template inclusive_scan<Op, T>(d_data, count, block_sums,
                                               stream);
        } else {
            status = block_scan_impl<BlockScan>::
                template exclusive_scan<Op, T>(d_data, count, block_sums,
                                               stream);
        }
        if (status != cudaSuccess) return status;

        status = scan_impl<BlockBased<ScanThenPropagate<BlockScan>>>::
            template exclusive_scan<Op, T>(block_sums, num_blocks,
                                           next_workspace,
                                           next_workspace_size, stream);
        if (status != cudaSuccess) return status;

        return block_scan_impl<BlockScan>::
            template add_block_offsets<Op, T>(d_data, count, block_sums,
                                              stream);
    }
};

} // namespace algo::cuda::scan::detail
