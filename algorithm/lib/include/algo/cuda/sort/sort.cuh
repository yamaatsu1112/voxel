#pragma once

#include <algo/cuda/scan/scan.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef __CUDACC__
#error "algo::cuda::sort requires CUDA compilation with nvcc"
#endif

#include <algo/cuda/sort/radix.cuh>
#include <cuda_runtime.h>

namespace algo::cuda::sort {

namespace detail {

using DefaultConfig =
    RadixSort<HistogramPass<SharedAtomicHistogram,
                            WarpLevelMultiSplitWarpRank, 4>,
              512, 4>;

template <class T>
inline constexpr bool kSupportedKeyType = std::is_same_v<T, std::uint32_t>;

template <class T>
inline constexpr bool kSupportedValueType = std::is_trivially_copyable_v<T>;

template <class> inline constexpr bool kAlwaysFalse = false;

template <class Config> struct sort_impl {
    static_assert(kAlwaysFalse<Config>,
                  "sort_impl is not implemented for this sort type");
};

template <class Config, class Key>
std::size_t validate_workspace(std::uint32_t count, void* workspace,
                               std::size_t workspace_size) {
    static_assert(kSupportedKeyType<Key>,
                  "sort currently only supports std::uint32_t keys");

    const std::size_t required =
        sort_impl<Config>::template required_workspace_size<Key>(count);
    if (required == 0)
        return 0;
    if (workspace == nullptr || workspace_size < required)
        return required;
    return 0;
}

template <class Config, class Key, class Value>
std::size_t validate_pairs_workspace(std::uint32_t count, void* workspace,
                                     std::size_t workspace_size) {
    static_assert(kSupportedKeyType<Key>,
                  "sort_pairs currently only supports std::uint32_t keys");
    static_assert(kSupportedValueType<Value>,
                  "sort_pairs requires trivially copyable values");

    const std::size_t required =
        sort_impl<Config>::template required_pairs_workspace_size<Key, Value>(
            count);
    if (required == 0)
        return 0;
    if (workspace == nullptr || workspace_size < required)
        return required;
    return 0;
}

template <class Config, class Key>
cudaError_t sort_keys_impl(Key* d_keys, std::uint32_t count, void* workspace,
                           std::size_t workspace_size, cudaStream_t stream) {
    const std::size_t missing =
        validate_workspace<Config, Key>(count, workspace, workspace_size);
    if (missing != 0)
        return cudaErrorInvalidValue;
    return sort_impl<Config>::template sort_keys<Key>(d_keys, count, workspace,
                                                      workspace_size, stream);
}

template <class Config, class Key, class Value>
cudaError_t sort_pairs_impl(Key* d_keys, Value* d_values, std::uint32_t count,
                            void* workspace, std::size_t workspace_size,
                            cudaStream_t stream) {
    const std::size_t missing = validate_pairs_workspace<Config, Key, Value>(
        count, workspace, workspace_size);
    if (missing != 0)
        return cudaErrorInvalidValue;
    return sort_impl<Config>::template sort_pairs<Key, Value>(
        d_keys, d_values, count, workspace, workspace_size, stream);
}

} // namespace detail

} // namespace algo::cuda::sort

namespace algo::cuda::sort {

template <class Config, class Key>
cudaError_t sort_keys(Key* d_keys, std::uint32_t count, void* d_workspace,
                      std::size_t workspace_size,
                      cudaStream_t stream = nullptr) {
    if (d_keys == nullptr && count != 0)
        return cudaErrorInvalidValue;
    return detail::sort_keys_impl<Config, Key>(d_keys, count, d_workspace,
                                               workspace_size, stream);
}

template <class Key>
cudaError_t sort_keys(Key* d_keys, std::uint32_t count, void* d_workspace,
                      std::size_t workspace_size,
                      cudaStream_t stream = nullptr) {
    return sort_keys<detail::DefaultConfig, Key>(d_keys, count, d_workspace,
                                                 workspace_size, stream);
}

template <class Config, class Key, class Value>
cudaError_t sort_pairs(Key* d_keys, Value* d_values, std::uint32_t count,
                       void* d_workspace, std::size_t workspace_size,
                       cudaStream_t stream = nullptr) {
    if ((d_keys == nullptr || d_values == nullptr) && count != 0) {
        return cudaErrorInvalidValue;
    }
    return detail::sort_pairs_impl<Config, Key, Value>(
        d_keys, d_values, count, d_workspace, workspace_size, stream);
}

template <class Key, class Value>
cudaError_t sort_pairs(Key* d_keys, Value* d_values, std::uint32_t count,
                       void* d_workspace, std::size_t workspace_size,
                       cudaStream_t stream = nullptr) {
    return sort_pairs<detail::DefaultConfig, Key, Value>(
        d_keys, d_values, count, d_workspace, workspace_size, stream);
}

template <class Config, class Key>
std::size_t required_workspace_size(std::uint32_t count) {
    static_assert(detail::kSupportedKeyType<Key>,
                  "sort currently only supports std::uint32_t keys");
    return detail::sort_impl<Config>::template required_workspace_size<Key>(
        count);
}

template <class Key> std::size_t required_workspace_size(std::uint32_t count) {
    return required_workspace_size<detail::DefaultConfig, Key>(count);
}

template <class Config, class Key, class Value>
std::size_t required_pairs_workspace_size(std::uint32_t count) {
    static_assert(detail::kSupportedKeyType<Key>,
                  "sort_pairs currently only supports std::uint32_t keys");
    static_assert(detail::kSupportedValueType<Value>,
                  "sort_pairs requires trivially copyable values");
    return detail::sort_impl<Config>::template required_pairs_workspace_size<
        Key, Value>(count);
}

template <class Key, class Value>
std::size_t required_pairs_workspace_size(std::uint32_t count) {
    return required_pairs_workspace_size<detail::DefaultConfig, Key, Value>(
        count);
}

} // namespace algo::cuda::sort
