#pragma once

#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/device_view.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {
namespace allocation::terminal_compact_all_depth {

inline constexpr std::uint32_t kTerminalMissingStateInheritedFilledShift = 31u;
inline constexpr std::uint32_t kTerminalMissingStateInheritedFilledBit =
    1u << kTerminalMissingStateInheritedFilledShift;
inline constexpr std::uint32_t kTerminalMissingStateDepthMask =
    ~kTerminalMissingStateInheritedFilledBit;
static_assert(kMaxDepth < kTerminalMissingStateInheritedFilledBit);

__device__ inline std::uint32_t
pack_terminal_missing_state(std::uint32_t first_missing_depth,
                            std::uint32_t inherited_filled) {
  return (first_missing_depth & kTerminalMissingStateDepthMask) |
         (inherited_filled != 0u ? kTerminalMissingStateInheritedFilledBit
                                 : 0u);
}

__device__ inline std::uint32_t
terminal_missing_state_first_missing_depth(std::uint32_t missing_state) {
  return missing_state & kTerminalMissingStateDepthMask;
}

__device__ inline std::uint32_t
terminal_missing_state_inherited_filled(std::uint32_t missing_state) {
  return (missing_state & kTerminalMissingStateInheritedFilledBit) >>
         kTerminalMissingStateInheritedFilledShift;
}

__device__ inline std::uint32_t
terminal_materialized_node_index(const DeviceGpuSvo &svo,
                                 const AllocationState *state,
                                 std::uint32_t rank) {
  return rank < state->base_free_node_count
             ? svo.free_node_indices[state->base_free_node_count - 1u - rank]
             : state->base_node_count + (rank - state->base_free_node_count);
}

__device__ inline std::uint32_t
terminal_materialized_leaf_index(const DeviceGpuSvo &svo,
                                 const AllocationState *state,
                                 std::uint32_t rank) {
  return rank < state->base_free_leaf_count
             ? svo.free_leaf_indices[state->base_free_leaf_count - 1u - rank]
             : state->base_leaf_count + (rank - state->base_free_leaf_count);
}

} // namespace allocation::terminal_compact_all_depth
} // namespace algo::svt::cuda::detail
