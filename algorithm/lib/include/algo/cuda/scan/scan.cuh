#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef __CUDACC__
#error "algo::cuda::scan requires CUDA compilation with nvcc"
#endif

#include <cuda_runtime.h>

#include <algo/cuda/scan/ops.cuh>
#include <algo/cuda/scan/block_based.cuh>
#include <algo/cuda/scan/fused.cuh>
#include <algo/cuda/scan/global.cuh>

namespace algo::cuda::scan {

namespace detail {

using DefaultConfig =
    BlockBased<DecoupledLookback<WarpShuffleBlock<512, 4>>>;

template <class T>
inline constexpr bool kSupportedValueType =
    std::is_integral_v<T> || std::is_floating_point_v<T>;

template <class>
inline constexpr bool kAlwaysFalse = false;

template <class Config>
struct scan_impl {
    static_assert(kAlwaysFalse<Config>,
                  "scan_impl is not implemented for this scan type");
};

template <class Config, class Op, class T>
std::size_t validate_workspace(std::uint32_t count, void* workspace,
                               std::size_t workspace_size) {
    static_assert(kSupportedValueType<T>,
                  "scan only supports integral and floating point types");

    const std::size_t required =
        scan_impl<Config>::template required_workspace_size<Op, T>(count);
    if (required == 0) return 0;
    if (workspace == nullptr || workspace_size < required) return required;
    return 0;
}

template <class Config, class Op, class T>
cudaError_t inclusive_scan_impl(T* d_data, std::uint32_t count, void* workspace,
                                std::size_t workspace_size,
                                cudaStream_t stream) {
    const std::size_t missing =
        validate_workspace<Config, Op, T>(count, workspace, workspace_size);
    if (missing != 0) return cudaErrorInvalidValue;
    return scan_impl<Config>::template inclusive_scan<Op, T>(
        d_data, count, workspace, workspace_size, stream);
}

template <class Config, class Op, class T>
cudaError_t exclusive_scan_impl(T* d_data, std::uint32_t count, void* workspace,
                                std::size_t workspace_size,
                                cudaStream_t stream) {
    const std::size_t missing =
        validate_workspace<Config, Op, T>(count, workspace, workspace_size);
    if (missing != 0) return cudaErrorInvalidValue;
    return scan_impl<Config>::template exclusive_scan<Op, T>(
        d_data, count, workspace, workspace_size, stream);
}

} // namespace detail

} // namespace algo::cuda::scan

namespace algo::cuda::scan {

template <class Config, class Op, class T>
cudaError_t inclusive_scan(T* d_data, std::uint32_t count, void* d_workspace,
                           std::size_t workspace_size,
                           cudaStream_t stream = nullptr) {
    if (d_data == nullptr && count != 0) return cudaErrorInvalidValue;
    return detail::inclusive_scan_impl<Config, Op, T>(
        d_data, count, d_workspace, workspace_size, stream);
}

template <class Op, class T>
cudaError_t inclusive_scan(T* d_data, std::uint32_t count, void* d_workspace,
                           std::size_t workspace_size,
                           cudaStream_t stream = nullptr) {
    return inclusive_scan<detail::DefaultConfig, Op, T>(
        d_data, count, d_workspace, workspace_size, stream);
}

template <class Config, class Op, class T>
cudaError_t exclusive_scan(T* d_data, std::uint32_t count, void* d_workspace,
                           std::size_t workspace_size,
                           cudaStream_t stream = nullptr) {
    if (d_data == nullptr && count != 0) return cudaErrorInvalidValue;
    return detail::exclusive_scan_impl<Config, Op, T>(
        d_data, count, d_workspace, workspace_size, stream);
}

template <class Op, class T>
cudaError_t exclusive_scan(T* d_data, std::uint32_t count, void* d_workspace,
                           std::size_t workspace_size,
                           cudaStream_t stream = nullptr) {
    return exclusive_scan<detail::DefaultConfig, Op, T>(
        d_data, count, d_workspace, workspace_size, stream);
}

template <class Config, class Op, class T>
std::size_t required_workspace_size(std::uint32_t count) {
    static_assert(detail::kSupportedValueType<T>,
                  "scan only supports integral and floating point types");
    return detail::scan_impl<Config>::
        template required_workspace_size<Op, T>(count);
}

template <class Config, class T>
cudaError_t inclusive_sum(T* d_data, std::uint32_t count, void* d_workspace,
                          std::size_t workspace_size,
                          cudaStream_t stream = nullptr) {
    return inclusive_scan<Config, Plus<T>, T>(d_data, count, d_workspace,
                                              workspace_size, stream);
}

template <class T>
cudaError_t inclusive_sum(T* d_data, std::uint32_t count, void* d_workspace,
                          std::size_t workspace_size,
                          cudaStream_t stream = nullptr) {
    return inclusive_sum<detail::DefaultConfig, T>(d_data, count, d_workspace,
                                                   workspace_size, stream);
}

template <class Config, class T>
cudaError_t exclusive_sum(T* d_data, std::uint32_t count, void* d_workspace,
                          std::size_t workspace_size,
                          cudaStream_t stream = nullptr) {
    return exclusive_scan<Config, Plus<T>, T>(d_data, count, d_workspace,
                                              workspace_size, stream);
}

template <class T>
cudaError_t exclusive_sum(T* d_data, std::uint32_t count, void* d_workspace,
                          std::size_t workspace_size,
                          cudaStream_t stream = nullptr) {
    return exclusive_sum<detail::DefaultConfig, T>(d_data, count, d_workspace,
                                                   workspace_size, stream);
}

template <class Config, class T>
std::size_t required_workspace_size(std::uint32_t count) {
    return required_workspace_size<Config, Plus<T>, T>(count);
}

template <class T>
std::size_t required_workspace_size(std::uint32_t count) {
    return required_workspace_size<detail::DefaultConfig, T>(count);
}

} // namespace algo::cuda::scan
