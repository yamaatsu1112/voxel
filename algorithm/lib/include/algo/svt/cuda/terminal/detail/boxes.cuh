#pragma once

#include <cstdint>

namespace algo::svt::cuda::detail {

struct TerminalBox {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
  std::int32_t size;
};

struct LeafBox {
  std::int32_t x0;
  std::int32_t x1;
  std::int32_t y0;
  std::int32_t y1;
  std::int32_t z0;
  std::int32_t z1;
};

} // namespace algo::svt::cuda::detail
