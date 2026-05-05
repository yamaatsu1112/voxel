#pragma once

#include <algo/svt/cuda/detail/edit_common.cuh>
#include <algo/svt/cuda/terminal/edit_config.cuh>

namespace algo::svt::cuda::detail {

template <class Allocation> struct terminal_allocation_impl {
  static_assert(
      kAlwaysFalse<Allocation>,
      "terminal_allocation_impl is not implemented for this allocation");
};

} // namespace algo::svt::cuda::detail
