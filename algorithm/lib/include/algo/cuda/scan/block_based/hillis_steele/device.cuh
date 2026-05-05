#pragma once

#include <algo/cuda/scan/block_based/hillis_steele/shared.cuh>
#include <algo/utils.hpp>

namespace algo::cuda::scan::detail {

template <int BlockSize, int ItemsPerThread, class T>
struct block_scan_shared_storage<HillisSteeleBlock<BlockSize, ItemsPerThread>,
                                 T> {
    T values[BlockSize * ItemsPerThread];
};

template <int BlockSize, int ItemsPerThread>
struct block_scan_device_impl<HillisSteeleBlock<BlockSize, ItemsPerThread>> {
    static_assert(::algo::is_power_of_two(BlockSize),
                  "Hillis-Steele shared scan requires power-of-two BlockSize");
    static_assert(::algo::is_power_of_two(ItemsPerThread),
                  "Hillis-Steele shared scan requires power-of-two ItemsPerThread");

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
        block_scan_shared_storage<HillisSteeleBlock<BlockSize, ItemsPerThread>,
                                  T>& shared_storage) {
        for (int item = 0; item < ItemsPerThread; ++item) {
            const std::uint32_t local_index = threadIdx.x + item * BlockSize;
            const std::uint32_t index = block_offset + local_index;
            const T value = index < count ? data[index] : Op::identity();
            shared_storage.values[local_index] = value;
        }
        __syncthreads();

        const T block_total =
            hillis_steele_block_inclusive_scan<Op, T, BlockSize,
                                               ItemsPerThread>(
                shared_storage.values);

        for (int item = 0; item < ItemsPerThread; ++item) {
            const std::uint32_t local_index = threadIdx.x + item * BlockSize;
            if constexpr (Inclusive) {
                items[item] = shared_storage.values[local_index];
            } else {
                items[item] = local_index == 0 ? Op::identity()
                                               : shared_storage.values[local_index - 1];
            }
        }

        return block_total;
    }
};

} // namespace algo::cuda::scan::detail
