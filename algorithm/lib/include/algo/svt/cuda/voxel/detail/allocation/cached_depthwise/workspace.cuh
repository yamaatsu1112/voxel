#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/allocation/types.cuh>
#include <algo/svt/cuda/detail/workspace_layout.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

namespace allocation::cached_depthwise {

struct Workspace {
    std::uint32_t* parent_indices;
    std::uint32_t* request_keys;
    std::uint32_t* unique_offsets;
    AllocationRequest* unique_requests;
    std::uint32_t* unique_count;
    AllocationState* allocation_state;
    void* request_scan_workspace;
    std::size_t request_scan_workspace_size;
};

inline std::size_t workspace_size(std::uint32_t count) {
    std::size_t offset = 0;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * count;
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
    offset += algo::cuda::scan::required_workspace_size<std::uint32_t>(count);
    return offset;
}

inline Workspace create_workspace(
    void* workspace, std::uint32_t count) {
    Workspace result{};
    std::size_t offset = 0;
    reserve_array(workspace, count, offset, result.parent_indices);
    reserve_array(workspace, count, offset, result.request_keys);
    reserve_array(workspace, count, offset, result.unique_offsets);
    reserve_array(workspace, count, offset, result.unique_requests);
    reserve_array(workspace, 1, offset, result.unique_count);
    reserve_array(workspace, 1, offset, result.allocation_state);

    result.request_scan_workspace_size =
        algo::cuda::scan::required_workspace_size<std::uint32_t>(count);
    reserve_workspace<std::max_align_t>(
        workspace, result.request_scan_workspace_size, offset,
        result.request_scan_workspace);
    return result;
}

} // namespace allocation::cached_depthwise

} // namespace algo::svt::cuda::detail
