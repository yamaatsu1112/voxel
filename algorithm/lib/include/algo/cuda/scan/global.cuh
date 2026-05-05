#pragma once

namespace algo::cuda::scan {

template <class Algorithm>
struct Global {
    using algorithm = Algorithm;
};

template <int BlockSize = 256>
struct BlellochGlobal {
    static_assert(BlockSize > 0, "BlellochGlobal requires BlockSize > 0");
    static constexpr int kBlockSize = BlockSize;
};

template <int BlockSize = 256>
struct HillisSteeleGlobal {
    static_assert(BlockSize > 0, "HillisSteeleGlobal requires BlockSize > 0");
    static constexpr int kBlockSize = BlockSize;
};

namespace detail {
template <class Config>
struct scan_impl;
} // namespace detail

} // namespace algo::cuda::scan

#include <algo/cuda/scan/global/blelloch/common.cuh>
#include <algo/cuda/scan/global/blelloch/exclusive.cuh>
#include <algo/cuda/scan/global/blelloch/inclusive.cuh>
#include <algo/cuda/scan/global/hillis_steele/kernels.cuh>
#include <algo/cuda/scan/global/blelloch.cuh>
#include <algo/cuda/scan/global/hillis_steele.cuh>
