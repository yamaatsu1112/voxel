#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/cuda/sort/sort.cuh>
#include <algo/svt/cuda/detail/workspace_layout.cuh>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

struct VoxelToLeafWorkspace {
    std::uint32_t* keys;
    std::uint32_t* local_bits;
    std::uint32_t* run_offsets;
    void* sort_workspace;
    std::size_t sort_workspace_size;
    void* scan_workspace;
    std::size_t scan_workspace_size;
};

inline std::size_t voxel_to_leaf_workspace_size(std::uint32_t count) {
    std::size_t offset = 0;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * count;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * count;
    offset = align_up<std::uint32_t>(offset);
    offset += sizeof(std::uint32_t) * count;
    offset = align_up<std::max_align_t>(offset);
    offset += algo::cuda::sort::required_pairs_workspace_size<std::uint32_t,
                                                              std::uint32_t>(
        count);
    offset = align_up<std::max_align_t>(offset);
    offset +=
        algo::cuda::scan::required_workspace_size_fused<std::uint32_t>(count);
    return offset;
}

inline VoxelToLeafWorkspace create_voxel_to_leaf_workspace(void* workspace,
                                                           std::uint32_t count) {
    VoxelToLeafWorkspace result{};
    std::size_t offset = 0;
    reserve_array(workspace, count, offset, result.keys);
    reserve_array(workspace, count, offset, result.local_bits);
    reserve_array(workspace, count, offset, result.run_offsets);
    result.sort_workspace_size =
        algo::cuda::sort::required_pairs_workspace_size<std::uint32_t,
                                                        std::uint32_t>(count);
    reserve_workspace<std::max_align_t>(workspace, result.sort_workspace_size,
                                        offset, result.sort_workspace);
    result.scan_workspace_size =
        algo::cuda::scan::required_workspace_size_fused<std::uint32_t>(count);
    reserve_workspace<std::max_align_t>(workspace, result.scan_workspace_size,
                                        offset, result.scan_workspace);
    return result;
}

} // namespace algo::svt::cuda::detail
