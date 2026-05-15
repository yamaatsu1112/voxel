#pragma once

#include <algo/cuda/sort/common.cuh>
#include <algo/cuda/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace algo::cuda::sort {

template <class... Values> struct value_arrays_t;

template <> struct value_arrays_t<> {};

template <class First, class... Rest> struct value_arrays_t<First, Rest...> {
    First* first = nullptr;
    value_arrays_t<Rest...> rest{};
};

inline value_arrays_t<> value_arrays() { return {}; }

template <class First, class... Rest>
value_arrays_t<First, Rest...> value_arrays(First* first, Rest*... rest) {
    return {first, value_arrays(rest...)};
}

namespace detail {

template <class... Values>
inline constexpr bool kSupportedValueArrayTypes =
    (std::is_trivially_copyable_v<Values> && ...);

inline bool all_value_array_pointers_valid(value_arrays_t<>) { return true; }

template <class First, class... Rest>
bool all_value_array_pointers_valid(value_arrays_t<First, Rest...> values) {
    return values.first != nullptr &&
           all_value_array_pointers_valid(values.rest);
}

template <class... Values>
std::size_t add_value_array_temp_storage_size(std::size_t offset,
                                              std::uint32_t count) {
    ((offset = align_up<Values>(offset),
      offset += sizeof(Values) * static_cast<std::size_t>(count)),
     ...);
    return offset;
}

template <class T>
T* allocate_temp_value(void* workspace, std::size_t& offset,
                       std::uint32_t count) {
    offset = align_up<T>(offset);
    T* ptr = pointer_at<T>(workspace, offset);
    offset += sizeof(T) * static_cast<std::size_t>(count);
    return ptr;
}

template <class... Values>
value_arrays_t<Values...> allocate_temp_value_arrays(void* workspace,
                                                     std::size_t& offset,
                                                     std::uint32_t count);

template <class... Values> struct value_array_temp_allocator;

template <> struct value_array_temp_allocator<> {
    static value_arrays_t<> allocate(void*, std::size_t&, std::uint32_t) {
        return {};
    }
};

template <class First, class... Rest>
struct value_array_temp_allocator<First, Rest...> {
    static value_arrays_t<First, Rest...> allocate(void* workspace,
                                                   std::size_t& offset,
                                                   std::uint32_t count) {
        First* first = allocate_temp_value<First>(workspace, offset, count);
        return {first,
                value_array_temp_allocator<Rest...>::allocate(workspace, offset,
                                                              count)};
    }
};

template <class... Values>
value_arrays_t<Values...> allocate_temp_value_arrays(void* workspace,
                                                     std::size_t& offset,
                                                     std::uint32_t count) {
    return value_array_temp_allocator<Values...>::allocate(workspace, offset,
                                                          count);
}

__device__ inline void copy_value_array_item(value_arrays_t<>, std::uint32_t,
                                             value_arrays_t<>, std::uint32_t) {}

template <class First, class... Rest>
__device__ void
copy_value_array_item(value_arrays_t<First, Rest...> output_values,
                      std::uint32_t output_index,
                      value_arrays_t<First, Rest...> input_values,
                      std::uint32_t input_index) {
    output_values.first[output_index] = input_values.first[input_index];
    copy_value_array_item(output_values.rest, output_index, input_values.rest,
                          input_index);
}

inline void swap_value_arrays(value_arrays_t<>&, value_arrays_t<>&) {}

template <class First, class... Rest>
void swap_value_arrays(value_arrays_t<First, Rest...>& lhs,
                       value_arrays_t<First, Rest...>& rhs) {
    First* const tmp = lhs.first;
    lhs.first = rhs.first;
    rhs.first = tmp;
    swap_value_arrays(lhs.rest, rhs.rest);
}

template <int BlockSize>
cudaError_t copy_value_array_buffers(value_arrays_t<>, value_arrays_t<>,
                                     std::uint32_t, cudaStream_t) {
    return cudaSuccess;
}

template <int BlockSize, class First, class... Rest>
cudaError_t copy_value_array_buffers(value_arrays_t<First, Rest...> dst,
                                     value_arrays_t<First, Rest...> src,
                                     std::uint32_t count,
                                     cudaStream_t stream) {
    cudaError_t status =
        ::algo::cuda::copy_buffer<First, BlockSize>(dst.first, src.first,
                                                   count, stream);
    if (status != cudaSuccess)
        return status;
    return copy_value_array_buffers<BlockSize>(dst.rest, src.rest, count,
                                              stream);
}

} // namespace detail

} // namespace algo::cuda::sort
