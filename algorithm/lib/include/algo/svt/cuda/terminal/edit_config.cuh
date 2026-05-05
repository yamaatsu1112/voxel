#pragma once

namespace algo::svt::cuda {

struct TerminalPlainDepthwiseAllocation {};
struct TerminalCompactAllDepthAllocation {};

struct TerminalDepthwiseRelease {};
struct TerminalFrontierRelease {};

template <class Allocation, class Release> struct TerminalEditConfig {
  using allocation = Allocation;
  using release = Release;
};

using DefaultTerminalEditConfig =
    TerminalEditConfig<TerminalCompactAllDepthAllocation,
                       TerminalFrontierRelease>;

} // namespace algo::svt::cuda
