#pragma once

#include <cstddef>

#include <algo/cuda/scan/block_based/device.cuh>

namespace algo::cuda::scan::detail {

inline constexpr unsigned int kDecoupledLookbackEmpty = 0;
inline constexpr unsigned int kDecoupledLookbackAggregate = 1;
inline constexpr unsigned int kDecoupledLookbackPrefix = 2;

template <class T> struct decoupled_lookback_record {
    T aggregate;
    T prefix;
    unsigned int state;
};

template <class T>
constexpr std::size_t decoupled_lookback_align_up(std::size_t offset) {
    constexpr std::size_t kAlign = alignof(T);
    return (offset + kAlign - 1) & ~(kAlign - 1);
}

template <class T>
T* decoupled_lookback_pointer_at(void* base, std::size_t offset) {
    return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <class T> struct decoupled_lookback_workspace {
    std::uint32_t* tile_counter = nullptr;
    decoupled_lookback_record<T>* records = nullptr;

    static std::size_t required_size(std::uint32_t num_blocks) {
        std::size_t offset = 0;
        offset = decoupled_lookback_align_up<std::uint32_t>(offset);
        offset += sizeof(std::uint32_t);
        offset =
            decoupled_lookback_align_up<decoupled_lookback_record<T>>(offset);
        offset += sizeof(decoupled_lookback_record<T>) *
                  static_cast<std::size_t>(num_blocks);
        return offset;
    }

    static decoupled_lookback_workspace create(void* workspace) {
        decoupled_lookback_workspace layout{};
        std::size_t offset = 0;
        offset = decoupled_lookback_align_up<std::uint32_t>(offset);
        layout.tile_counter =
            decoupled_lookback_pointer_at<std::uint32_t>(workspace, offset);
        offset += sizeof(std::uint32_t);
        offset =
            decoupled_lookback_align_up<decoupled_lookback_record<T>>(offset);
        layout.records =
            decoupled_lookback_pointer_at<decoupled_lookback_record<T>>(
                workspace, offset);
        return layout;
    }
};

template <class T>
__device__ __forceinline__ unsigned int
decoupled_lookback_load_state(volatile decoupled_lookback_record<T>* records,
                              std::uint32_t index) {
    auto* state = const_cast<unsigned int*>(&records[index].state);
    return atomicAdd(state, 0u);
}

template <class T>
__device__ __forceinline__ void decoupled_lookback_publish_aggregate(
    volatile decoupled_lookback_record<T>* records, std::uint32_t index,
    T aggregate) {
    records[index].aggregate = aggregate;
    __threadfence();
    auto* state = const_cast<unsigned int*>(&records[index].state);
    atomicExch(state, kDecoupledLookbackAggregate);
}

template <class T>
__device__ __forceinline__ void decoupled_lookback_publish_prefix(
    volatile decoupled_lookback_record<T>* records, std::uint32_t index,
    T prefix) {
    records[index].prefix = prefix;
    __threadfence();
    auto* state = const_cast<unsigned int*>(&records[index].state);
    atomicExch(state, kDecoupledLookbackPrefix);
}

template <class Op, class T>
__device__ T decoupled_lookback_prefix(
    volatile decoupled_lookback_record<T>* records, std::uint32_t block_index) {
    T prefix = Op::identity();
    const Op op{};
    std::uint32_t cursor = block_index;

    while (cursor > 0) {
        --cursor;

        unsigned int state = kDecoupledLookbackEmpty;
        while (state == kDecoupledLookbackEmpty) {
            state = decoupled_lookback_load_state(records, cursor);
        }

        if (state == kDecoupledLookbackPrefix) {
            prefix = op(records[cursor].prefix, prefix);
            break;
        }

        // if state == kDecoupledLookbackAggregate
        prefix = op(records[cursor].aggregate, prefix);
    }

    return prefix;
}

template <bool Inclusive, class BlockScan, class Op, class T>
__global__ void decoupled_lookback_kernel(
    T* data, std::uint32_t count, decoupled_lookback_record<T>* records,
    std::uint32_t* tile_counter) {
    using device_impl = block_scan_device_impl<BlockScan>;

    constexpr std::uint32_t kTileSize = device_impl::kTileSize;

    __shared__ block_scan_shared_storage<BlockScan, T> shared_storage;
    __shared__ T block_prefix;
    __shared__ std::uint32_t tile_index;

    if (threadIdx.x == 0)
        tile_index = atomicAdd(tile_counter, 1u);
    __syncthreads();

    const std::uint32_t block_offset = tile_index * kTileSize;
    T items[device_impl::kItemsPerThread];
    const T block_aggregate =
        device_impl::template scan_tile<Inclusive, Op, T>(
            data, count, block_offset, items, shared_storage);
    if (threadIdx.x == 0) {
        const Op op{};
        decoupled_lookback_publish_aggregate(records, tile_index,
                                             block_aggregate);
        const T prefix = decoupled_lookback_prefix<Op>(records, tile_index);
        block_prefix = prefix;
        decoupled_lookback_publish_prefix(records, tile_index,
                                          op(prefix, block_aggregate));
    }
    __syncthreads();

    const T offset = block_prefix;
    const Op op{};
    for (int item = 0; item < device_impl::kItemsPerThread; ++item) {
        const std::uint32_t global_index =
            device_impl::global_index(block_offset, item);
        if (global_index < count)
            data[global_index] = op(offset, items[item]);
    }
}

template <class BlockScan>
struct scan_impl<BlockBased<DecoupledLookback<BlockScan>>> {
    using device_impl = block_scan_device_impl<BlockScan>;
    static constexpr std::uint32_t kTileSize = device_impl::kTileSize;

    template <class Op, class T>
    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= kTileSize)
            return 0;
        const std::uint32_t num_blocks = ::algo::ceil_div(count, kTileSize);
        return decoupled_lookback_workspace<T>::required_size(num_blocks);
    }

    template <class Op, class T>
    static cudaError_t
    inclusive_scan(T* d_data, std::uint32_t count, void* workspace,
                   std::size_t workspace_size, cudaStream_t stream) {
        return run<true, Op, T>(d_data, count, workspace, workspace_size,
                                stream);
    }

    template <class Op, class T>
    static cudaError_t
    exclusive_scan(T* d_data, std::uint32_t count, void* workspace,
                   std::size_t workspace_size, cudaStream_t stream) {
        return run<false, Op, T>(d_data, count, workspace, workspace_size,
                                 stream);
    }

  private:
    template <bool Inclusive, class Op, class T>
    static cudaError_t run(T* d_data, std::uint32_t count, void* workspace,
                           std::size_t workspace_size, cudaStream_t stream) {
        (void)workspace_size;
        if (count == 0)
            return cudaSuccess;
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
        const std::size_t required = required_workspace_size<Op, T>(count);
        const auto layout = decoupled_lookback_workspace<T>::create(workspace);
        const cudaError_t status =
            cudaMemsetAsync(workspace, 0, required, stream);
        if (status != cudaSuccess)
            return status;

        auto* records = layout.records;
        auto* tile_counter = layout.tile_counter;
        decoupled_lookback_kernel<Inclusive, BlockScan, Op, T>
            <<<num_blocks, device_impl::kBlockSize, 0, stream>>>(d_data, count,
                                                                 records,
                                                                 tile_counter);
        return cudaGetLastError();
    }
};

} // namespace algo::cuda::scan::detail
