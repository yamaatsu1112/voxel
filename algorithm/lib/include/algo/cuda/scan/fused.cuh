#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>

#include <algo/cuda/scan/ops.cuh>
#include <algo/cuda/scan/block_based/decoupled_lookback.cuh>

namespace algo::cuda::scan::detail {

using DefaultFusedBlockScan = WarpShuffleBlock<256, 4>;

template <class T>
inline constexpr bool kSupportedFusedValueType =
    std::is_integral_v<T> || std::is_floating_point_v<T>;

template <class Input, class Transform>
using fused_scan_value_t = std::remove_cv_t<std::remove_reference_t<decltype(
    Transform::run(std::declval<std::uint32_t>(),
                   std::declval<const Input&>()))>>;

template <class>
inline constexpr bool kAlwaysFalseFused = false;

template <class BlockScan> struct fused_block_scan_impl {
    static_assert(kAlwaysFalseFused<BlockScan>,
                  "fused_block_scan_impl is not implemented for this block "
                  "scan type");
};

template <int BlockSize, int ItemsPerThread>
struct fused_block_scan_impl<WarpShuffleBlock<BlockSize, ItemsPerThread>> {
    using block_scan = WarpShuffleBlock<BlockSize, ItemsPerThread>;
    using device_impl = block_scan_device_impl<block_scan>;

    static constexpr int kBlockSize = device_impl::kBlockSize;
    static constexpr int kItemsPerThread = device_impl::kItemsPerThread;
    static constexpr std::uint32_t kTileSize = device_impl::kTileSize;

    template <bool Inclusive, class Op, class T>
    __device__ __forceinline__ static T scan_tile(
        T (&items)[ItemsPerThread],
        block_scan_shared_storage<block_scan, T>& shared_storage) {
        constexpr int kWarpCount = BlockSize / kWarpSize;

        const int lane = threadIdx.x & (kWarpSize - 1);
        const int warp_id = threadIdx.x / kWarpSize;

        const Op op{};
        T thread_total = Op::identity();
        for (int item = 0; item < ItemsPerThread; ++item)
            thread_total = op(thread_total, items[item]);

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

template <bool Inclusive, class Op, class T, class Input, class Transform,
          class PostScan>
__global__ void decoupled_lookback_fused_kernel(
    Input input, std::uint32_t count, decoupled_lookback_record<T>* records) {
    using fused_impl = fused_block_scan_impl<DefaultFusedBlockScan>;

    constexpr std::uint32_t kTileSize = fused_impl::kTileSize;

    __shared__ block_scan_shared_storage<DefaultFusedBlockScan, T>
        shared_storage;
    __shared__ T block_prefix;

    const std::uint32_t block_offset = blockIdx.x * kTileSize;
    T transformed[fused_impl::kItemsPerThread];
    T prefixes[fused_impl::kItemsPerThread];

    for (int item = 0; item < fused_impl::kItemsPerThread; ++item) {
        const std::uint32_t global_index =
            fused_impl::device_impl::global_index(block_offset, item);
        const T value = global_index < count
                            ? Transform::run(global_index, input)
                            : Op::identity();
        transformed[item] = value;
        prefixes[item] = value;
    }

    const T block_aggregate =
        fused_impl::template scan_tile<Inclusive, Op, T>(prefixes,
                                                         shared_storage);

    if (threadIdx.x == 0) {
        const Op op{};
        T prefix = Op::identity();
        if (records != nullptr) {
            decoupled_lookback_publish_aggregate(records, blockIdx.x,
                                                 block_aggregate);
            if (blockIdx.x != 0) {
                prefix = decoupled_lookback_prefix<Op>(
                    records, static_cast<std::uint32_t>(blockIdx.x));
            }
            decoupled_lookback_publish_prefix(records, blockIdx.x,
                                              op(prefix, block_aggregate));
        }
        block_prefix = prefix;
    }
    __syncthreads();

    const T offset = block_prefix;
    const Op op{};
    for (int item = 0; item < fused_impl::kItemsPerThread; ++item) {
        const std::uint32_t global_index =
            fused_impl::device_impl::global_index(block_offset, item);
        if (global_index < count) {
            PostScan::run(global_index, input, transformed[item],
                          op(offset, prefixes[item]));
        }
    }
}

template <class Op, class T>
std::size_t required_fused_workspace_size(std::uint32_t count) {
    static_assert(kSupportedFusedValueType<T>,
                  "fused scan only supports integral and floating point types");

    using fused_impl = fused_block_scan_impl<DefaultFusedBlockScan>;
    if (count <= fused_impl::kTileSize)
        return 0;
    const std::uint32_t num_blocks = ::algo::ceil_div(count, fused_impl::kTileSize);
    return sizeof(decoupled_lookback_record<T>) *
           static_cast<std::size_t>(num_blocks);
}

template <class Op, class T>
std::size_t validate_fused_workspace(std::uint32_t count, void* workspace,
                                     std::size_t workspace_size) {
    const std::size_t required = required_fused_workspace_size<Op, T>(count);
    if (required == 0)
        return 0;
    if (workspace == nullptr || workspace_size < required)
        return required;
    return 0;
}

template <bool Inclusive, class Op, class Transform, class PostScan,
          class Input>
cudaError_t fused_scan_impl(Input input, std::uint32_t count, void* workspace,
                            std::size_t workspace_size, cudaStream_t stream) {
    using T = fused_scan_value_t<Input, Transform>;
    static_assert(kSupportedFusedValueType<T>,
                  "fused scan only supports integral and floating point types");

    const std::size_t missing =
        validate_fused_workspace<Op, T>(count, workspace, workspace_size);
    if (missing != 0)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;

    using fused_impl = fused_block_scan_impl<DefaultFusedBlockScan>;
    auto* records = static_cast<decoupled_lookback_record<T>*>(workspace);
    const std::size_t required = required_fused_workspace_size<Op, T>(count);
    if (required != 0) {
        const cudaError_t status =
            cudaMemsetAsync(records, 0, required, stream);
        if (status != cudaSuccess)
            return status;
    }

    const std::uint32_t num_blocks = ::algo::ceil_div(count, fused_impl::kTileSize);
    decoupled_lookback_fused_kernel<Inclusive, Op, T, Input, Transform,
                                    PostScan>
        <<<num_blocks, fused_impl::kBlockSize, 0, stream>>>(input, count,
                                                            records);
    return cudaGetLastError();
}

} // namespace algo::cuda::scan::detail

namespace algo::cuda::scan {

template <class ScanValue>
std::size_t required_workspace_size_fused(std::uint32_t count) {
    return detail::required_fused_workspace_size<Plus<ScanValue>, ScanValue>(
        count);
}

template <class Op, class ScanValue>
std::size_t required_workspace_size_fused(std::uint32_t count) {
    return detail::required_fused_workspace_size<Op, ScanValue>(count);
}

template <class Op, class Transform, class PostScan, class Input>
cudaError_t inclusive_scan_fused(Input input, std::uint32_t count,
                                 void* d_workspace,
                                 std::size_t workspace_size,
                                 cudaStream_t stream = nullptr) {
    return detail::fused_scan_impl<true, Op, Transform, PostScan>(
        input, count, d_workspace, workspace_size, stream);
}

template <class Op, class Transform, class PostScan, class Input>
cudaError_t exclusive_scan_fused(Input input, std::uint32_t count,
                                 void* d_workspace,
                                 std::size_t workspace_size,
                                 cudaStream_t stream = nullptr) {
    return detail::fused_scan_impl<false, Op, Transform, PostScan>(
        input, count, d_workspace, workspace_size, stream);
}

template <class Transform, class PostScan, class Input>
cudaError_t inclusive_sum_fused(Input input, std::uint32_t count,
                                void* d_workspace,
                                std::size_t workspace_size,
                                cudaStream_t stream = nullptr) {
    using T = detail::fused_scan_value_t<Input, Transform>;
    return detail::fused_scan_impl<true, Plus<T>, Transform, PostScan>(
        input, count, d_workspace, workspace_size, stream);
}

template <class Transform, class PostScan, class Input>
cudaError_t exclusive_sum_fused(Input input, std::uint32_t count,
                                void* d_workspace,
                                std::size_t workspace_size,
                                cudaStream_t stream = nullptr) {
    using T = detail::fused_scan_value_t<Input, Transform>;
    return detail::fused_scan_impl<false, Plus<T>, Transform, PostScan>(
        input, count, d_workspace, workspace_size, stream);
}

} // namespace algo::cuda::scan
