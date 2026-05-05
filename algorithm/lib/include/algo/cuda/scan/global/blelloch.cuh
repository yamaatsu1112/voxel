#pragma once

namespace algo::cuda::scan::detail {

using blelloch_inclusive_impl_t = blelloch_from_exclusive_inclusive;

template <int BlockSize>
struct scan_impl<Global<BlellochGlobal<BlockSize>>> {
    template <class Op, class T>
    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= 1) return 0;
        return sizeof(T) *
               static_cast<std::size_t>(::algo::next_power_of_two(count));
    }

    template <class Op, class T>
    static cudaError_t inclusive_scan(T* d_data, std::uint32_t count,
                                      void* workspace, std::size_t,
                                      cudaStream_t stream) {
        return blelloch_inclusive_impl<blelloch_inclusive_impl_t>::
            template run<Op, T, BlockSize>(d_data, count, workspace, stream);
    }

    template <class Op, class T>
    static cudaError_t exclusive_scan(T* d_data, std::uint32_t count,
                                      void* workspace, std::size_t,
                                      cudaStream_t stream) {
        if (count == 0) return cudaSuccess;
        auto* buffer = static_cast<T*>(workspace);
        const std::uint32_t padded_count = ::algo::next_power_of_two(count);
        cudaError_t status =
            fill_padded<Op, T, BlockSize>(buffer, d_data, count, padded_count,
                                          stream);
        if (status != cudaSuccess) return status;
        status = blelloch_exclusive<Op, T, BlockSize>(buffer, padded_count,
                                                      stream);
        if (status != cudaSuccess) return status;
        return ::algo::cuda::copy_buffer<T, BlockSize>(d_data, buffer, count,
                                                       stream);
    }
};

} // namespace algo::cuda::scan::detail
