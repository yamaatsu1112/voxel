#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/cuda/sort/value_arrays.cuh>

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
inline constexpr bool kSupportedKeyType = radix_key_traits<T>::kSupported;

template <class> inline constexpr bool kAlwaysFalse = false;

template <class Config> struct sort_impl {
    static_assert(kAlwaysFalse<Config>,
                  "sort_impl is not implemented for this sort type");
};

template <class Config, class Key>
std::size_t validate_workspace(std::uint32_t count, void* workspace,
                               std::size_t workspace_size) {
    static_assert(kSupportedKeyType<Key>,
                  "sort key type is not supported");

    const std::size_t required =
        sort_impl<Config>::template required_workspace_size<Key>(count);
    if (required == 0)
        return 0;
    if (workspace == nullptr || workspace_size < required)
        return required;
    return 0;
}

template <class Config, class Key, class... Values>
std::size_t validate_by_key_workspace(std::uint32_t count, void* workspace,
                                      std::size_t workspace_size) {
    static_assert(kSupportedKeyType<Key>,
                  "sort_by_key key type is not supported");
    static_assert(kSupportedValueArrayTypes<Values...>,
                  "sort_by_key requires trivially copyable values");

    const std::size_t required =
        sort_impl<Config>::template required_sort_by_key_workspace_size<
            Key, Values...>(count);
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

template <class Config, class Key, class... Values>
cudaError_t sort_by_key_impl(Key* d_keys, value_arrays_t<Values...> d_values,
                             std::uint32_t count, void* workspace,
                             std::size_t workspace_size,
                             cudaStream_t stream) {
    const std::size_t missing =
        validate_by_key_workspace<Config, Key, Values...>(
            count, workspace, workspace_size);
    if (missing != 0)
        return cudaErrorInvalidValue;
    return sort_impl<Config>::template sort_by_key<Key, Values...>(
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

template <class Config, class Key, class... Values>
cudaError_t sort_by_key(Key* d_keys, value_arrays_t<Values...> d_values,
                        std::uint32_t count, void* d_workspace,
                        std::size_t workspace_size,
                        cudaStream_t stream = nullptr) {
    if (d_keys == nullptr && count != 0)
        return cudaErrorInvalidValue;
    if (count != 0 && !detail::all_value_array_pointers_valid(d_values))
        return cudaErrorInvalidValue;
    return detail::sort_by_key_impl<Config, Key, Values...>(
        d_keys, d_values, count, d_workspace, workspace_size, stream);
}

template <class Key, class... Values>
cudaError_t sort_by_key(Key* d_keys, value_arrays_t<Values...> d_values,
                        std::uint32_t count, void* d_workspace,
                        std::size_t workspace_size,
                        cudaStream_t stream = nullptr) {
    return sort_by_key<detail::DefaultConfig, Key, Values...>(
        d_keys, d_values, count, d_workspace, workspace_size, stream);
}

template <class Config, class Key>
std::size_t required_workspace_size(std::uint32_t count) {
    static_assert(detail::kSupportedKeyType<Key>,
                  "sort key type is not supported");
    return detail::sort_impl<Config>::template required_workspace_size<Key>(
        count);
}

template <class Key> std::size_t required_workspace_size(std::uint32_t count) {
    return required_workspace_size<detail::DefaultConfig, Key>(count);
}

template <class Config, class Key, class... Values>
std::size_t required_sort_by_key_workspace_size(std::uint32_t count) {
    static_assert(detail::kSupportedKeyType<Key>,
                  "sort_by_key key type is not supported");
    static_assert(detail::kSupportedValueArrayTypes<Values...>,
                  "sort_by_key requires trivially copyable values");
    return detail::sort_impl<Config>::
        template required_sort_by_key_workspace_size<Key, Values...>(count);
}

template <class Key, class... Values>
std::size_t required_sort_by_key_workspace_size(
    std::uint32_t count, value_arrays_t<Values...>) {
    return required_sort_by_key_workspace_size<detail::DefaultConfig, Key,
                                               Values...>(count);
}

} // namespace algo::cuda::sort
