#pragma once

namespace algo::cuda::scan::detail {

template <class ReducePolicy>
struct reduce_then_scan_reduce_impl;

template <class Op, class T, int BlockSize, int ItemsPerThread>
__global__ void block_tile_reduce_kernel(const T* data, std::uint32_t count,
                                         T* block_sums) {
    __shared__ T shared[BlockSize];
    const Op op{};

    const std::uint32_t tile_start =
        static_cast<std::uint32_t>(blockIdx.x) * BlockSize * ItemsPerThread;

    T thread_sum = Op::identity();
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const std::uint32_t index =
            tile_start + static_cast<std::uint32_t>(threadIdx.x) +
            static_cast<std::uint32_t>(item) * BlockSize;
        if (index < count) thread_sum = op(thread_sum, data[index]);
    }

    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    std::uint32_t active = BlockSize;
    while (active > 1) {
        const std::uint32_t next_active = (active + 1) >> 1;
        if (static_cast<std::uint32_t>(threadIdx.x) < next_active) {
            const std::uint32_t rhs =
                static_cast<std::uint32_t>(threadIdx.x) + next_active;
            if (rhs < active)
                shared[threadIdx.x] = op(shared[threadIdx.x], shared[rhs]);
        }
        __syncthreads();
        active = next_active;
    }

    if (threadIdx.x == 0) block_sums[blockIdx.x] = shared[0];
}

template <>
struct reduce_then_scan_reduce_impl<BlockTileReduction> {
    template <class BlockScan, class Op, class T>
    static constexpr std::size_t required_workspace_size(std::uint32_t) {
        return 0;
    }

    template <class BlockScan, class Op, class T>
    static cudaError_t reduce(const T* d_input, std::uint32_t count,
                              T* d_block_sums, void*, std::size_t,
                              cudaStream_t stream) {
        constexpr std::uint32_t kTileSize = block_scan_impl<BlockScan>::kTileSize;
        constexpr int kBlockSize = BlockScan::kBlockSize;
        constexpr int kItemsPerThread = BlockScan::kItemsPerThread;
        const auto grid =
            static_cast<unsigned int>(::algo::ceil_div(count, kTileSize));
        block_tile_reduce_kernel<Op, T, kBlockSize, kItemsPerThread>
            <<<grid, kBlockSize, 0, stream>>>(d_input, count, d_block_sums);
        return cudaGetLastError();
    }
};

template <class BlockScan, class ReducePolicy>
struct scan_impl<BlockBased<ReduceThenScan<BlockScan, ReducePolicy>>> {
    template <class Op, class T>
    static std::size_t required_workspace_size(std::uint32_t count) {
        constexpr std::uint32_t kTileSize = block_scan_impl<BlockScan>::kTileSize;
        if (count <= kTileSize) return 0;
        const std::uint32_t num_blocks = ::algo::ceil_div(count, kTileSize);
        const std::size_t block_sums_bytes =
            sizeof(T) * static_cast<std::size_t>(num_blocks);
        const std::size_t reduce_workspace_bytes =
            reduce_then_scan_reduce_impl<ReducePolicy>::
                template required_workspace_size<BlockScan, Op, T>(count);
        return block_sums_bytes + reduce_workspace_bytes +
               scan_impl<BlockBased<ReduceThenScan<BlockScan, ReducePolicy>>>::
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
        const std::size_t reduce_workspace_bytes =
            reduce_then_scan_reduce_impl<ReducePolicy>::
                template required_workspace_size<BlockScan, Op, T>(count);
        auto* block_sums = static_cast<T*>(workspace);
        auto* reduce_workspace = static_cast<void*>(
            static_cast<std::byte*>(workspace) + block_sums_bytes);
        auto* next_workspace = static_cast<void*>(
            static_cast<std::byte*>(workspace) + block_sums_bytes +
            reduce_workspace_bytes);
        const std::size_t next_workspace_size =
            workspace_size - block_sums_bytes - reduce_workspace_bytes;

        cudaError_t status =
            reduce_then_scan_reduce_impl<ReducePolicy>::
                template reduce<BlockScan, Op, T>(
                    d_data, count, block_sums, reduce_workspace,
                    reduce_workspace_bytes, stream);
        if (status != cudaSuccess) return status;

        status =
            scan_impl<BlockBased<ReduceThenScan<BlockScan, ReducePolicy>>>::
                template exclusive_scan<Op, T>(block_sums, num_blocks,
                                               next_workspace,
                                               next_workspace_size, stream);
        if (status != cudaSuccess) return status;

        if constexpr (Inclusive) {
            status = block_scan_impl<BlockScan>::
                template inclusive_scan<Op, T>(d_data, count, nullptr, stream);
        } else {
            status = block_scan_impl<BlockScan>::
                template exclusive_scan<Op, T>(d_data, count, nullptr, stream);
        }
        if (status != cudaSuccess) return status;

        return block_scan_impl<BlockScan>::
            template add_block_offsets<Op, T>(d_data, count, block_sums,
                                              stream);
    }
};

} // namespace algo::cuda::scan::detail
