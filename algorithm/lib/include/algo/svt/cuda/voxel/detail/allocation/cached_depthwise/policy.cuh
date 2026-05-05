#pragma once

#include <algo/svt/cuda/voxel/detail/allocation/cached_depthwise/workspace.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/common.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace algo::svt::cuda::detail {

template <> struct allocation_impl<CachedDepthwiseAllocation> {
    using Workspace = allocation::cached_depthwise::Workspace;

    static std::size_t workspace_size(std::uint32_t count) {
        return allocation::cached_depthwise::workspace_size(count);
    }

    static Workspace create_workspace(void* workspace, std::uint32_t count) {
        return allocation::cached_depthwise::create_workspace(workspace, count);
    }

    static cudaError_t allocate_paths(DeviceGpuSvo svo,
                                      const LeafMask* leaf_masks,
                                      const std::uint32_t* leaf_count,
                                      std::uint32_t leaf_mask_capacity,
                                      Workspace& workspace,
                                      cudaStream_t stream);
};

} // namespace algo::svt::cuda::detail
