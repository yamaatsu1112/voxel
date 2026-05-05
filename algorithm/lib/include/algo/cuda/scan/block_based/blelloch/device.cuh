#pragma once

#include <algo/cuda/scan/block_based/blelloch/shared.cuh>
#include <algo/cuda/scan/block_based/layout.cuh>
#include <algo/utils.hpp>

namespace algo::cuda::scan::detail {

template <int BlockSize, int ItemsPerThread, class Layout, class T>
struct block_scan_shared_storage<
    BlellochBlock<BlockSize, ItemsPerThread, Layout>, T> {
    static constexpr std::uint32_t kTileSize = BlockSize * ItemsPerThread;
    using layout_impl = shared_layout_impl<Layout, T>;

    T values[layout_impl::storage_size(kTileSize)];
};

template <int BlockSize, int ItemsPerThread, class Layout>
struct block_scan_device_impl<
    BlellochBlock<BlockSize, ItemsPerThread, Layout>> {
    static_assert(::algo::is_power_of_two(BlockSize),
                  "Blelloch shared scan requires power-of-two BlockSize");
    static_assert(::algo::is_power_of_two(ItemsPerThread),
                  "Blelloch shared scan requires power-of-two ItemsPerThread");

    static constexpr int kBlockSize = BlockSize;
    static constexpr int kItemsPerThread = ItemsPerThread;
    static constexpr std::uint32_t kTileSize = BlockSize * ItemsPerThread;

    __device__ __forceinline__ static std::uint32_t
    global_index(std::uint32_t block_offset, int item) {
        return block_offset + threadIdx.x + item * BlockSize;
    }

    template <bool Inclusive, class Op, class T>
    __device__ __forceinline__ static T scan_tile(
        T* data, std::uint32_t count, std::uint32_t block_offset,
        T (&items)[ItemsPerThread],
        block_scan_shared_storage<
            BlellochBlock<BlockSize, ItemsPerThread, Layout>, T>&
            shared_storage) {
        using layout_impl = shared_layout_impl<Layout, T>;

        for (int item = 0; item < ItemsPerThread; ++item) {
            const std::uint32_t local_index = threadIdx.x + item * BlockSize;
            const std::uint32_t index = block_offset + local_index;
            const T value = index < count ? data[index] : Op::identity();
            if constexpr (Inclusive) {
                items[item] = value;
            }
            shared_storage.values[layout_impl::map(local_index)] = value;
        }
        __syncthreads();

        const T block_total =
            blelloch_block_exclusive_scan_impl<Layout, Op, T, BlockSize,
                                               ItemsPerThread>(
                shared_storage.values);

        for (int item = 0; item < ItemsPerThread; ++item) {
            const std::uint32_t local_index = threadIdx.x + item * BlockSize;
            const T exclusive =
                shared_storage.values[layout_impl::map(local_index)];
            if constexpr (Inclusive) {
                items[item] = Op{}(exclusive, items[item]);
            } else {
                items[item] = exclusive;
            }
        }

        return block_total;
    }
};

} // namespace algo::cuda::scan::detail
