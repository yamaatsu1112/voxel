#pragma once

namespace algo::svt::cuda {

enum class EditOp {
    Place,
    Destroy,
};

namespace detail {

template <class> inline constexpr bool kAlwaysFalse = false;

} // namespace detail
} // namespace algo::svt::cuda
