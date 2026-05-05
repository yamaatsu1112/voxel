#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace algo::svt::detail {

struct SVDAGBuildNode {
    std::array<uint32_t, 8> child = {};
};

struct SVDAGBuildSortEntry {
    SVDAGBuildNode key;
    uint32_t old_index = 0;
};

[[nodiscard]] inline bool has_uniform_children(const SVDAGBuildNode& node) {
    return std::all_of(node.child.begin() + 1, node.child.end(),
                       [&node](uint32_t child) {
                           return child == node.child[0];
                       });
}

[[nodiscard]] inline bool has_uniform_children(const SVDAGBuildNode& node,
                                               uint32_t value) {
    return std::all_of(node.child.begin(), node.child.end(),
                       [value](uint32_t child) { return child == value; });
}

} // namespace algo::svt::detail
