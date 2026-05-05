#pragma once

#include "../cuda_test_utils.cuh"

#include <algo/svt/cuda/svo.cuh>
#include <algo/svt/cuda/voxel/edit.cuh>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using cuda_test::copy_from_device;
using cuda_test::copy_scalar_from_device;
using cuda_test::copy_to_device;
using cuda_test::device_alloc;
using cuda_test::has_cuda_device;
using cuda_test::sync_cuda;

using TestGpuSvo = algo::svt::cuda::GpuSvo<256, 64>;

template <class Svo>
algo::svt::cuda::GpuSvoCounters counters_for(const Svo& svo) {
    return copy_scalar_from_device(svo.view().counters);
}

std::uint32_t expected_leaf_key(std::uint32_t x, std::uint32_t y,
                                std::uint32_t z) {
    using namespace algo::svt::cuda;
    return make_leaf_key(x >> kLeafVoxelCountExp, y >> kLeafVoxelCountExp,
                         z >> kLeafVoxelCountExp);
}

__global__ void query_voxels_kernel(algo::svt::cuda::DeviceGpuSvo svo,
                                    const algo::svt::cuda::VoxelEdit* queries,
                                    std::uint8_t* results,
                                    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const auto query = queries[index];
    results[index] =
        algo::svt::cuda::get_voxel(svo, query.x, query.y, query.z) ? 1u : 0u;
}

} // namespace
