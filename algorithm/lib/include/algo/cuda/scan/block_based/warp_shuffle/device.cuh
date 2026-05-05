#pragma once

#include <algo/cuda/scan/block_based/warp_shuffle/primitives.cuh>

namespace algo::cuda::scan::detail {

template <int BlockSize, int ItemsPerThread, class T>
struct block_scan_shared_storage<WarpShuffleBlock<BlockSize, ItemsPerThread>,
                                 T> {
    T warp_totals[BlockSize / kWarpSize];
};

template <int BlockSize, int ItemsPerThread>
struct block_scan_device_impl<WarpShuffleBlock<BlockSize, ItemsPerThread>> {
    static_assert(BlockSize > 0, "WarpShuffleBlock requires BlockSize > 0");
    static_assert(BlockSize % 32 == 0,
                  "WarpShuffleBlock requires BlockSize to be a multiple of 32");
    static_assert(BlockSize <= 1024,
                  "WarpShuffleBlock requires BlockSize <= 1024");
    static_assert(ItemsPerThread > 0,
                  "WarpShuffleBlock requires ItemsPerThread > 0");

    static constexpr int kBlockSize = BlockSize;
    static constexpr int kItemsPerThread = ItemsPerThread;
    static constexpr std::uint32_t kTileSize = BlockSize * ItemsPerThread;

    __device__ __forceinline__ static std::uint32_t
    global_index(std::uint32_t block_offset, int item) {
        return block_offset + threadIdx.x * ItemsPerThread + item;
    }

    template <bool Inclusive, class Op, class T>
    __device__ __forceinline__ static T scan_tile(
        T* data, std::uint32_t count, std::uint32_t block_offset,
        T (&items)[ItemsPerThread],
        block_scan_shared_storage<WarpShuffleBlock<BlockSize, ItemsPerThread>,
                                  T>& shared_storage) {
        constexpr int kWarpCount = BlockSize / kWarpSize;

        const int lane = threadIdx.x & (kWarpSize - 1);
        const int warp_id = threadIdx.x / kWarpSize;

        const Op op{};
        T thread_total = Op::identity();
        for (int item = 0; item < ItemsPerThread; ++item) {
            const std::uint32_t index = global_index(block_offset, item);
            const T value = index < count ? data[index] : Op::identity();
            items[item] = value;
            thread_total = op(thread_total, value);
        }

        const T thread_inclusive_prefix =
            warp_inclusive_scan<Op>(thread_total);
        if (lane == kWarpSize - 1)
            shared_storage.warp_totals[warp_id] = thread_inclusive_prefix;
        __syncthreads();

        if (warp_id == 0) {
            const T warp_total =
                lane < kWarpCount ? shared_storage.warp_totals[lane]
                                  : Op::identity();
            const T warp_total_prefix = warp_inclusive_scan<Op>(warp_total);
            if (lane < kWarpCount)
                shared_storage.warp_totals[lane] = warp_total_prefix;
        }
        __syncthreads();

        const T warp_offset =
            warp_id == 0 ? Op::identity()
                         : shared_storage.warp_totals[warp_id - 1];
        const auto thread_inclusive_prefix_carrier =
            to_warp_carrier(thread_inclusive_prefix);
        const auto previous_thread_prefix_carrier =
            __shfl_up_sync(kWarpFullMask, thread_inclusive_prefix_carrier, 1);
        const T thread_exclusive_prefix =
            lane == 0 ? Op::identity()
                      : from_warp_carrier<T>(previous_thread_prefix_carrier);
        const T thread_offset_prefix = op(warp_offset, thread_exclusive_prefix);

        T running = thread_offset_prefix;
        for (int item = 0; item < ItemsPerThread; ++item) {
            const T value = items[item];
            if constexpr (Inclusive) {
                running = op(running, value);
                items[item] = running;
            } else {
                items[item] = running;
                running = op(running, value);
            }
        }

        return shared_storage.warp_totals[kWarpCount - 1];
    }
};

} // namespace algo::cuda::scan::detail
