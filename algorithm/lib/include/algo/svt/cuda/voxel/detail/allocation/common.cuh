#pragma once

#include <algo/svt/cuda/detail/edit_common.cuh>
#include <algo/svt/cuda/voxel/edit_config.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

namespace algo::svt::cuda::detail {

template <class Allocation> struct allocation_impl {
    static_assert(kAlwaysFalse<Allocation>,
                  "allocation_impl is not implemented for this allocation");
};

} // namespace algo::svt::cuda::detail
