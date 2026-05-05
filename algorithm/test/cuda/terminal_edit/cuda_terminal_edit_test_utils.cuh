#pragma once

#include "../cuda_test_utils.cuh"

#include <algo/svt/cuda/query.cuh>
#include <algo/svt/cuda/svo.cuh>
#include <algo/svt/cuda/terminal/edit.cuh>
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

algo::svt::cuda::GpuSvoCounters counters_for(const TestGpuSvo& svo) {
    return copy_scalar_from_device(svo.view().counters);
}

struct VoxelQuery {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
};

std::uint64_t leaf_key(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return algo::svt::cuda::make_leaf_key(x, y, z);
}

__global__ void query_voxels_kernel(algo::svt::cuda::DeviceGpuSvo svo,
                                    const VoxelQuery* queries,
                                    std::uint8_t* results,
                                    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const auto query = queries[index];
    results[index] =
        algo::svt::cuda::get_voxel(svo, query.x, query.y, query.z) ? 1u : 0u;
}

std::vector<std::uint8_t>
query_voxels(TestGpuSvo& svo, const std::vector<VoxelQuery>& queries) {
    auto* d_queries = device_alloc<VoxelQuery>(queries.size());
    auto* d_results = device_alloc<std::uint8_t>(queries.size());
    EXPECT_NE(d_queries, nullptr);
    EXPECT_NE(d_results, nullptr);
    copy_to_device(d_queries, queries);
    query_voxels_kernel<<<1, 64>>>(svo.view(), d_queries, d_results,
                                   static_cast<std::uint32_t>(queries.size()));
    sync_cuda();
    std::vector<std::uint8_t> results =
        copy_from_device(d_results, queries.size());
    cudaFree(d_results);
    cudaFree(d_queries);
    return results;
}

void expect_cuda_device_or_skip() {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }
}

template <class Config = algo::svt::cuda::DefaultTerminalEditConfig>
void apply_terminal_nodes(
    TestGpuSvo& svo,
    const std::vector<algo::svt::cuda::TerminalNodeInput>& nodes,
    algo::svt::cuda::EditOp op) {
    auto* d_nodes =
        device_alloc<algo::svt::cuda::TerminalNodeInput>(nodes.size());
    ASSERT_NE(d_nodes, nullptr);
    copy_to_device(d_nodes, nodes);

    const auto workspace_size =
        algo::svt::cuda::detail::apply_terminal_edits_workspace_size<
            typename Config::allocation, typename Config::release>(
            static_cast<std::uint32_t>(nodes.size()), 0u,
            static_cast<std::uint32_t>(nodes.size() *
                                       algo::svt::cuda::kGroupSize));
    auto* workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    const auto status =
        algo::svt::cuda::detail::apply_terminal_edits<
            typename Config::allocation, typename Config::release>(
            svo.view(), d_nodes, static_cast<std::uint32_t>(nodes.size()),
            nullptr, 0u, op, workspace, workspace_size);
    ASSERT_EQ(status, cudaSuccess);
    sync_cuda();

    cudaFree(workspace);
    cudaFree(d_nodes);
}

template <class Config = algo::svt::cuda::DefaultTerminalEditConfig>
void apply_terminal_leaves(
    TestGpuSvo& svo,
    const std::vector<algo::svt::cuda::TerminalLeafInput>& leaves,
    algo::svt::cuda::EditOp op) {
    auto* d_leaves =
        device_alloc<algo::svt::cuda::TerminalLeafInput>(leaves.size());
    ASSERT_NE(d_leaves, nullptr);
    copy_to_device(d_leaves, leaves);

    const auto workspace_size =
        algo::svt::cuda::detail::apply_terminal_edits_workspace_size<
            typename Config::allocation, typename Config::release>(
            0u, static_cast<std::uint32_t>(leaves.size()), 8u);
    auto* workspace = device_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    const auto status =
        algo::svt::cuda::detail::apply_terminal_edits<
            typename Config::allocation, typename Config::release>(
            svo.view(), nullptr, 0u, d_leaves,
            static_cast<std::uint32_t>(leaves.size()), op, workspace,
            workspace_size);
    ASSERT_EQ(status, cudaSuccess);
    sync_cuda();

    cudaFree(workspace);
    cudaFree(d_leaves);
}

} // namespace
