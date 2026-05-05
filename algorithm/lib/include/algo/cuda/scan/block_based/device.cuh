#pragma once

#include <algo/utils.hpp>

#include <cstdint>

namespace algo::cuda::scan::detail {

template <class BlockScan, class T>
struct block_scan_shared_storage;

template <class BlockScan>
struct block_scan_device_impl;

} // namespace algo::cuda::scan::detail

#include <algo/cuda/scan/block_based/blelloch/device.cuh>
#include <algo/cuda/scan/block_based/hillis_steele/device.cuh>
#include <algo/cuda/scan/block_based/warp_shuffle/device.cuh>

namespace algo::cuda::scan::detail {

template <bool Inclusive, class BlockScan, class Op, class T>
__global__ void block_scan_kernel(T* data, std::uint32_t count, T* block_sums) {
    using device_impl = block_scan_device_impl<BlockScan>;

    __shared__ block_scan_shared_storage<BlockScan, T> shared_storage;

    const std::uint32_t block_offset = blockIdx.x * device_impl::kTileSize;
    T items[device_impl::kItemsPerThread];
    const T aggregate = device_impl::template scan_tile<Inclusive, Op, T>(
        data, count, block_offset, items, shared_storage);

    for (int item = 0; item < device_impl::kItemsPerThread; ++item) {
        const std::uint32_t global_index =
            device_impl::global_index(block_offset, item);
        if (global_index < count)
            data[global_index] = items[item];
    }

    if (block_sums != nullptr && threadIdx.x == 0)
        block_sums[blockIdx.x] = aggregate;
}

template <class BlockScan, class Op, class T>
__global__ void block_scan_add_block_offsets_kernel(
    T* data, std::uint32_t count, const T* block_offsets) {
    using device_impl = block_scan_device_impl<BlockScan>;

    const std::uint32_t block_offset = blockIdx.x * device_impl::kTileSize;
    const T offset = block_offsets[blockIdx.x];
    const Op op{};

    for (int item = 0; item < device_impl::kItemsPerThread; ++item) {
        const std::uint32_t global_index =
            device_impl::global_index(block_offset, item);
        if (global_index < count)
            data[global_index] = op(offset, data[global_index]);
    }
}

template <class BlockScan, class Op, class T>
inline cudaError_t block_scan_add_block_offsets(
    T* data, std::uint32_t count, const T* block_offsets, cudaStream_t stream) {
    using device_impl = block_scan_device_impl<BlockScan>;

    if (count == 0)
        return cudaSuccess;

    const auto grid = static_cast<unsigned int>(
        ::algo::ceil_div(count, device_impl::kTileSize));
    block_scan_add_block_offsets_kernel<BlockScan, Op, T>
        <<<grid, device_impl::kBlockSize, 0, stream>>>(
            data, count, block_offsets);
    return cudaGetLastError();
}

} // namespace algo::cuda::scan::detail
