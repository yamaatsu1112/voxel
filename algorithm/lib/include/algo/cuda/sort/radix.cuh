#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/cuda/sort/uint_key.cuh>

#include <cstddef>
#include <cstdint>

#ifndef __CUDACC__
#error "algo::cuda::sort::radix requires CUDA compilation with nvcc"
#endif

#include <cuda_runtime.h>

namespace algo::cuda::sort {

struct BucketWiseScan {};
struct FlattenedScan {};

template <class ScanPolicy = BucketWiseScan> struct FlagPrefixSumPass {
    using scan_policy = ScanPolicy;
};

struct SharedAtomicHistogram {};
struct WarpBallotHistogram {};
struct WarpLevelMultiSplitHistogram {};

struct BucketBallotWarpRank {};
struct WarpLevelMultiSplitWarpRank {};

struct WarpBallotBlockHistogram {};
struct SharedAtomicGlobalOffsetsBlockHistogram {};

template <class HistogramPolicy, class WarpRankPolicy, int ItemsPerThread>
struct HistogramPass {
    static_assert(ItemsPerThread > 0,
                  "HistogramPass requires ItemsPerThread > 0");
    using histogram_policy = HistogramPolicy;
    using warp_rank_policy = WarpRankPolicy;
    static constexpr int kItemsPerThread = ItemsPerThread;
};

template <class BlockHistogramPolicy, class LocalRankPolicy,
          class GlobalOffsetsBlockHistogramPolicy, int ItemsPerThread>
struct OneSweepPass {
    static_assert(ItemsPerThread > 0,
                  "OneSweepPass requires ItemsPerThread > 0");
    using block_histogram_policy = BlockHistogramPolicy;
    using local_rank_policy = LocalRankPolicy;
    using global_offsets_block_histogram_policy =
        GlobalOffsetsBlockHistogramPolicy;
    static constexpr int kItemsPerThread = ItemsPerThread;
};

template <class PassPolicy, int BlockSize = 256, int RadixBits = 4,
          int KeyBits = 32>
struct RadixSort {
    using pass_policy = PassPolicy;
    static_assert(BlockSize > 0, "RadixSort requires BlockSize > 0");
    static_assert(RadixBits > 0, "RadixSort requires RadixBits > 0");
    static_assert(RadixBits < 32, "RadixSort requires RadixBits < 32");
    static_assert(KeyBits > 0, "RadixSort requires KeyBits > 0");
    static_assert(KeyBits % RadixBits == 0,
                  "RadixSort requires KeyBits to be a multiple of RadixBits");
    static constexpr int kBlockSize = BlockSize;
    static constexpr int kRadixBits = RadixBits;
    static constexpr int kKeyBits = KeyBits;
};

} // namespace algo::cuda::sort

#include <algo/cuda/sort/radix/common.cuh>
#include <algo/cuda/sort/radix/flag_prefix_sum/bucket_wise/policy.cuh>
#include <algo/cuda/sort/radix/flag_prefix_sum/flattened/policy.cuh>
#include <algo/cuda/sort/radix/flag_prefix_sum/radix.cuh>
#include <algo/cuda/sort/radix/histogram/policy.cuh>
#include <algo/cuda/sort/radix/onesweep/policy.cuh>
