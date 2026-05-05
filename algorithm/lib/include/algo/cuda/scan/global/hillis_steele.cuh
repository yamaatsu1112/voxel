#pragma once

namespace algo::cuda::scan::detail {

template <int BlockSize>
struct scan_impl<Global<HillisSteeleGlobal<BlockSize>>> {
    template <class Op, class T>
    static std::size_t required_workspace_size(std::uint32_t count) {
        if (count <= 1) return 0;
        return sizeof(T) * static_cast<std::size_t>(count);
    }

    template <class Op, class T>
    static cudaError_t inclusive_scan(T* d_data, std::uint32_t count,
                                      void* workspace, std::size_t,
                                      cudaStream_t stream) {
        if (count <= 1) return cudaSuccess;
        auto* buffer = static_cast<T*>(workspace);
        T* src = d_data;
        T* dst = buffer;
        for (std::uint32_t offset = 1; offset < count; offset <<= 1) {
            cudaError_t status =
                hillis_steele_step<Op, T, BlockSize>(dst, src, count, offset,
                                                     stream);
            if (status != cudaSuccess) return status;
            T* temp = src;
            src = dst;
            dst = temp;
        }
        if (src != d_data) {
            return ::algo::cuda::copy_buffer<T, BlockSize>(d_data, src, count,
                                                           stream);
        }
        return cudaSuccess;
    }

    template <class Op, class T>
    static cudaError_t exclusive_scan(T* d_data, std::uint32_t count,
                                      void* workspace, std::size_t,
                                      cudaStream_t stream) {
        if (count == 0) return cudaSuccess;
        auto* buffer = static_cast<T*>(workspace);
        cudaError_t status =
            init_shifted_exclusive<Op, T, BlockSize>(buffer, d_data, count,
                                                     stream);
        if (status != cudaSuccess) return status;
        if (count == 1) {
            return ::algo::cuda::copy_buffer<T, BlockSize>(d_data, buffer,
                                                           count, stream);
        }

        T* src = buffer;
        T* dst = d_data;
        for (std::uint32_t offset = 1; offset < count; offset <<= 1) {
            status = hillis_steele_step<Op, T, BlockSize>(dst, src, count,
                                                          offset, stream);
            if (status != cudaSuccess) return status;
            T* temp = src;
            src = dst;
            dst = temp;
        }
        if (src != d_data) {
            return ::algo::cuda::copy_buffer<T, BlockSize>(d_data, src, count,
                                                           stream);
        }
        return cudaSuccess;
    }
};

} // namespace algo::cuda::scan::detail
