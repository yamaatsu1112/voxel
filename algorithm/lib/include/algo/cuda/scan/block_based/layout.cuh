#pragma once

#include <algo/cuda/utils.cuh>
#include <algo/utils.hpp>

namespace algo::cuda::scan::detail {

template <class Layout, class T>
struct shared_layout_impl;

template <class T>
struct shared_layout_impl<DirectSharedLayout, T> {
    __host__ __device__ static constexpr std::uint32_t map(
        std::uint32_t index) {
        return index;
    }

    __host__ __device__ static constexpr std::uint32_t storage_size(
        std::uint32_t logical_size) {
        return logical_size;
    }
};

template <class T>
struct shared_layout_impl<PaddedSharedLayout, T> {
    static constexpr std::uint32_t kBankCount = 32;
    static constexpr std::uint32_t kBankWidthBytes = 4;
    static constexpr std::uint32_t kBankSpanBytes = kBankCount * kBankWidthBytes;

    __host__ __device__ static constexpr std::uint32_t padding(
        std::uint32_t index) {
        return static_cast<std::uint32_t>(
            (static_cast<std::size_t>(index) * sizeof(T)) / kBankSpanBytes);
    }

    __host__ __device__ static constexpr std::uint32_t map(
        std::uint32_t index) {
        return index + padding(index);
    }

    __host__ __device__ static constexpr std::uint32_t storage_size(
        std::uint32_t logical_size) {
        return logical_size == 0 ? 0 : (map(logical_size - 1) + 1);
    }
};
} // namespace algo::cuda::scan::detail
