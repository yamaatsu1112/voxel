#pragma once

#include <cstddef>
#include <cstdint>

namespace algo::cuda::scan {

struct DirectSharedLayout {};
struct PaddedSharedLayout {};

struct BlockTileReduction {};

template <class BlockScan>
struct ScanThenPropagate {
    using block_scan = BlockScan;
};

template <class BlockScan, class ReducePolicy = BlockTileReduction>
struct ReduceThenScan {
    using block_scan = BlockScan;
    using reduce_policy = ReducePolicy;
};

template <class BlockScan>
struct DecoupledLookback {
    using block_scan = BlockScan;
};

template <class BlockAlgorithm>
struct BlockBased {
    using block_algorithm = BlockAlgorithm;
};

template <int BlockSize = 256, int ItemsPerThread = 2,
          class Layout = DirectSharedLayout>
struct BlellochBlock {
    static_assert(BlockSize > 0, "BlellochBlock requires BlockSize > 0");
    static_assert(ItemsPerThread > 0,
                  "BlellochBlock requires ItemsPerThread > 0");
    static constexpr int kBlockSize = BlockSize;
    static constexpr int kItemsPerThread = ItemsPerThread;
    using layout = Layout;
};

template <int BlockSize = 256, int ItemsPerThread = 2>
struct HillisSteeleBlock {
    static_assert(BlockSize > 0, "HillisSteeleBlock requires BlockSize > 0");
    static_assert(ItemsPerThread > 0,
                  "HillisSteeleBlock requires ItemsPerThread > 0");
    static constexpr int kBlockSize = BlockSize;
    static constexpr int kItemsPerThread = ItemsPerThread;
};

template <int BlockSize = 256, int ItemsPerThread = 4>
struct WarpShuffleBlock {
    static_assert(BlockSize > 0, "WarpShuffleBlock requires BlockSize > 0");
    static_assert(ItemsPerThread > 0,
                  "WarpShuffleBlock requires ItemsPerThread > 0");
    static constexpr int kBlockSize = BlockSize;
    static constexpr int kItemsPerThread = ItemsPerThread;
};

namespace detail {
template <class Config>
struct scan_impl;
} // namespace detail

} // namespace algo::cuda::scan

#include <algo/cuda/scan/block_based/common.cuh>
#include <algo/cuda/scan/block_based/layout.cuh>
#include <algo/cuda/scan/block_based/device.cuh>
#include <algo/cuda/scan/block_based/blelloch/impl.cuh>
#include <algo/cuda/scan/block_based/hillis_steele/impl.cuh>
#include <algo/cuda/scan/block_based/warp_shuffle/impl.cuh>
#include <algo/cuda/scan/block_based/decoupled_lookback.cuh>
#include <algo/cuda/scan/block_based/scan_then_propagate.cuh>
#include <algo/cuda/scan/block_based/reduce_then_scan.cuh>
