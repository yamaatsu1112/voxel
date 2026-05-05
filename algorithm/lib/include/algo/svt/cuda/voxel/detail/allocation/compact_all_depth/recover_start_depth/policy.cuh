#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/common.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/recover_start_depth/workspace.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace algo::svt::cuda::detail {

template <class InitMode, class ScheduleMode>
struct allocation_impl<
    CompactAllDepthAllocation<InitMode, RecoverStartDepth, ScheduleMode>> {
    using Workspace =
        allocation::compact_all_depth::recover_start_depth::Workspace;

    static std::size_t workspace_size(std::uint32_t count) {
        return allocation::compact_all_depth::recover_start_depth::
            workspace_size(
                count,
                std::is_same_v<ScheduleMode, Nodewise<ExplicitRequests>>);
    }

    static Workspace create_workspace(void* workspace, std::uint32_t count) {
        return allocation::compact_all_depth::recover_start_depth::
            create_workspace(
                workspace, count,
                std::is_same_v<ScheduleMode, Nodewise<ExplicitRequests>>);
    }

    static cudaError_t allocate_paths(DeviceGpuSvo svo,
                                      const LeafMask* leaf_masks,
                                      const std::uint32_t* leaf_count,
                                      std::uint32_t leaf_mask_capacity,
                                      Workspace& workspace,
                                      cudaStream_t stream);
};

} // namespace algo::svt::cuda::detail
