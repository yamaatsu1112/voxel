#pragma once

#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/device_view.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

inline constexpr std::uint32_t kMaterializeRequestFilledShift = 31u;
inline constexpr std::uint32_t kMaterializeRequestFilledBit =
    1u << kMaterializeRequestFilledShift;
inline constexpr std::uint32_t kMaterializeRequestDepthMask =
    ~kMaterializeRequestFilledBit;
static_assert(kMaxDepth < kMaterializeRequestFilledBit);

inline constexpr std::uint32_t kMaterializeParentRefIsRequestBit = 1u << 0u;

inline constexpr std::uint32_t kMissingStateInheritedFilledShift = 31u;
inline constexpr std::uint32_t kMissingStateInheritedFilledBit =
    1u << kMissingStateInheritedFilledShift;
inline constexpr std::uint32_t kMissingStateDepthMask =
    ~kMissingStateInheritedFilledBit;
static_assert(kMaxDepth < kMissingStateInheritedFilledBit);

__device__ inline std::uint32_t pack_materialize_request_depth(
    std::uint32_t depth, std::uint32_t filled) {
    return (depth & kMaterializeRequestDepthMask) |
           (filled != 0u ? kMaterializeRequestFilledBit : 0u);
}

template <class Request>
__device__ inline std::uint32_t materialize_request_depth(
    const Request& request) {
    return request.packed_depth & kMaterializeRequestDepthMask;
}

template <class Request>
__device__ inline bool materialize_request_filled(const Request& request) {
    return (request.packed_depth & kMaterializeRequestFilledBit) != 0u;
}

__device__ inline std::uint32_t materialize_request_metadata(
    bool parent_ref_is_request) {
    return parent_ref_is_request ? kMaterializeParentRefIsRequestBit : 0u;
}

template <class Request>
__device__ inline bool materialize_request_parent_ref_is_request(
    const Request& request) {
    return (request.metadata & kMaterializeParentRefIsRequestBit) != 0u;
}

__device__ inline std::uint32_t pack_missing_state(
    std::uint32_t first_missing_depth, std::uint32_t inherited_filled) {
    return (first_missing_depth & kMissingStateDepthMask) |
           (inherited_filled != 0u ? kMissingStateInheritedFilledBit : 0u);
}

__device__ inline std::uint32_t missing_state_first_missing_depth(
    std::uint32_t missing_state) {
    return missing_state & kMissingStateDepthMask;
}

__device__ inline std::uint32_t missing_state_inherited_filled(
    std::uint32_t missing_state) {
    return (missing_state & kMissingStateInheritedFilledBit) >>
           kMissingStateInheritedFilledShift;
}

__device__ inline std::uint32_t materialized_node_index(
    const DeviceGpuSvo& svo, const AllocationState* state, std::uint32_t rank) {
    return rank < state->base_free_node_count
               ? svo.free_node_indices[state->base_free_node_count - 1u - rank]
               : state->base_node_count + (rank - state->base_free_node_count);
}

__device__ inline std::uint32_t materialized_leaf_index(
    const DeviceGpuSvo& svo, const AllocationState* state, std::uint32_t rank) {
    return rank < state->base_free_leaf_count
               ? svo.free_leaf_indices[state->base_free_leaf_count - 1u - rank]
               : state->base_leaf_count + (rank - state->base_free_leaf_count);
}

__device__ inline bool materialize_slot_emits(
    std::uint32_t leaf_key, std::uint32_t previous_leaf_key,
    bool has_previous_leaf, std::uint32_t first_missing_depth,
    std::uint32_t depth) {
    if (first_missing_depth == 0u || depth < first_missing_depth)
        return false;

    const std::uint32_t prefix = prefix_for_leaf_key(leaf_key, depth);
    return !has_previous_leaf ||
           prefix_for_leaf_key(previous_leaf_key, depth) != prefix;
}

} // namespace algo::svt::cuda::detail
