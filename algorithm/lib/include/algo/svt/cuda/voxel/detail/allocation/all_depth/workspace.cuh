#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/materialize_common.cuh>
#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/detail/workspace_layout.cuh>
#include <algo/svt/cuda/device_view.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::all_depth {

struct MaterializeRequest {
    std::uint32_t packed_depth;
    std::uint32_t prefix;
    std::uint32_t parent_ref;
    std::uint32_t metadata;
};

static_assert(sizeof(MaterializeRequest) == 16);

struct Workspace {
    std::uint32_t* missing_states;
    std::uint32_t* existing_parent_indices;
    std::uint32_t* materialize_offsets;
    MaterializeRequest* materialize_requests;
    std::uint32_t* materialize_count;
    std::uint32_t* materialize_node_count;

    std::uint32_t* request_keys;
    std::uint32_t* unique_offsets;
    AllocationRequest* unique_requests;
    std::uint32_t* unique_count;
    AllocationState* allocation_state;

    void* request_scan_workspace;
    std::size_t request_scan_workspace_size;
};

inline std::uint32_t materialize_candidate_count(std::uint32_t count) {
    return count * kMaxDepth;
}

inline std::size_t workspace_size(std::uint32_t count) {
    const std::uint32_t candidate_count = materialize_candidate_count(count);
    const std::size_t scan_count =
        std::max<std::size_t>(candidate_count, count);

    std::size_t offset = 0;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * count;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * count;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * candidate_count;
    offset = align_up<MaterializeRequest>(offset);
    offset += sizeof(MaterializeRequest) * candidate_count;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t);
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t);

    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * count;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * count;
    offset = align_up<AllocationRequest>(offset);
    offset += sizeof(AllocationRequest) * count;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t);
    offset = align_up<AllocationState>(offset);
    offset += sizeof(AllocationState);

    offset = align_up<std::max_align_t>(offset);
    offset += algo::cuda::scan::required_workspace_size<std::uint32_t>(
        static_cast<std::uint32_t>(scan_count));
    return offset;
}

inline Workspace create_workspace(
    void* workspace, std::uint32_t count) {
    const std::uint32_t candidate_count = materialize_candidate_count(count);
    const std::uint32_t scan_count =
        candidate_count > count ? candidate_count : count;

    Workspace result{};
    std::size_t offset = 0;
    reserve_array(workspace, count, offset, result.missing_states);
    reserve_array(workspace, count, offset, result.existing_parent_indices);
    reserve_array(workspace, candidate_count, offset,
                  result.materialize_offsets);
    reserve_array(workspace, candidate_count, offset,
                  result.materialize_requests);
    reserve_array(workspace, 1, offset, result.materialize_count);
    reserve_array(workspace, 1, offset, result.materialize_node_count);

    reserve_array(workspace, count, offset, result.request_keys);
    reserve_array(workspace, count, offset, result.unique_offsets);
    reserve_array(workspace, count, offset, result.unique_requests);
    reserve_array(workspace, 1, offset, result.unique_count);
    reserve_array(workspace, 1, offset, result.allocation_state);

    result.request_scan_workspace_size =
        algo::cuda::scan::required_workspace_size<std::uint32_t>(scan_count);
    reserve_workspace<std::max_align_t>(
        workspace, result.request_scan_workspace_size, offset,
        result.request_scan_workspace);
    return result;
}

} // namespace allocation::all_depth

} // namespace algo::svt::cuda::detail
