#pragma once

#include <cstdint>

namespace algo::svt::cuda::detail {

struct AllocationRequest {
    std::uint32_t node_index;
    std::uint32_t child_index;
};

struct AllocationState {
    std::uint32_t base_node_count;
    std::uint32_t base_leaf_count;
    std::uint32_t base_free_node_count;
    std::uint32_t base_free_leaf_count;
};

static_assert(sizeof(AllocationRequest) == 8);

} // namespace algo::svt::cuda::detail
