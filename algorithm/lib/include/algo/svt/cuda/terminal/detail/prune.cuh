#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/launch.cuh>
#include <algo/svt/cuda/terminal/detail/workspace/edit.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

__host__ __device__ inline std::uint32_t
terminal_cell_interval_end(const CellWriteRequest& request) {
    return (request.prefix + 1u)
           << (kGroupSizeExp * (kMaxDepth - request.level));
}

__global__ void fill_terminal_cell_end_kernel(
    const TerminalRequest* requests, std::uint32_t request_count,
    std::uint32_t* cell_ends) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= request_count)
        return;

    const TerminalRequest request = requests[index];
    cell_ends[index] = terminal_request_is_cell(request)
                           ? terminal_cell_interval_end(
                                 terminal_request_cell(request))
                           : 0u;
}

__device__ inline std::uint32_t
terminal_prune_group_end(const std::uint32_t* keys, std::uint32_t index,
                         std::uint32_t request_count) {
    const std::uint32_t key = keys[index];
    while (index + 1u < request_count && keys[index + 1u] == key)
        ++index;
    return index + 1u;
}

__global__ void mark_pruned_terminal_requests_kernel(
    const TerminalRequest* requests, const std::uint32_t* request_keys,
    const std::uint32_t* coverage_before, std::uint32_t request_count,
    std::uint32_t* emit_flags) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= request_count)
        return;

    const bool group_begin =
        index == 0u || request_keys[index - 1u] != request_keys[index];
    if (!group_begin) {
        // Duplicate keys are handled by the first request in the sorted group.
        emit_flags[index] = 0u;
        return;
    }
    const std::uint32_t group_end =
        terminal_prune_group_end(request_keys, index, request_count);

    std::uint32_t group_max_cell_end = 0u;
    bool has_brick = false;
    for (std::uint32_t cursor = index; cursor < group_end; ++cursor) {
        const TerminalRequest request = requests[cursor];
        if (terminal_request_is_cell(request)) {
            const std::uint32_t end =
                terminal_cell_interval_end(terminal_request_cell(request));
            if (end > group_max_cell_end)
                group_max_cell_end = end;
        } else {
            has_brick = true;
        }
    }

    // A kept cell covers the group's representative leaf, so any same-key brick
    // request is redundant. Otherwise, merge same-key bricks unless prior cells
    // already cover the leaf.
    const bool keep_cell = group_max_cell_end > coverage_before[index];
    const bool keep_brick =
        !keep_cell && has_brick
        && request_keys[index] >= coverage_before[index];
    emit_flags[index] = (keep_cell || keep_brick) ? 1u : 0u;
}

__global__ void compact_pruned_terminal_requests_kernel(
    const TerminalRequest* requests, const std::uint32_t* request_keys,
    const std::uint32_t* coverage_before, const std::uint32_t* emit_prefixes,
    std::uint32_t request_count, TerminalRequest* pruned_requests) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= request_count)
        return;

    const std::uint32_t output_index =
        index == 0u ? 0u : emit_prefixes[index - 1u];
    const std::uint32_t emit_count = emit_prefixes[index] - output_index;
    if (emit_count == 0u)
        return;

    const std::uint32_t group_end =
        terminal_prune_group_end(request_keys, index, request_count);
    std::uint32_t group_max_cell_end = 0u;
    CellWriteRequest group_max_cell{};
    std::uint64_t brick_mask = 0u;
    bool has_brick = false;
    for (std::uint32_t cursor = index; cursor < group_end; ++cursor) {
        const TerminalRequest candidate = requests[cursor];
        if (terminal_request_is_cell(candidate)) {
            const CellWriteRequest cell = terminal_request_cell(candidate);
            const std::uint32_t end =
                terminal_cell_interval_end(cell);
            if (end > group_max_cell_end) {
                group_max_cell_end = end;
                group_max_cell = cell;
            }
        } else {
            brick_mask |= terminal_request_brick(candidate).mask64;
            has_brick = true;
        }
    }
    const bool keep_cell = group_max_cell_end > coverage_before[index];
    const bool keep_brick =
        !keep_cell && has_brick
        && request_keys[index] >= coverage_before[index];

    if (keep_cell) {
        pruned_requests[output_index] =
            make_terminal_cell_request(group_max_cell);
    } else if (keep_brick) {
        pruned_requests[output_index] = make_terminal_brick_request(
            {request_keys[index], brick_mask});
    }
}

inline cudaError_t prune_terminal_requests(const TerminalRequest* requests,
                                           TerminalRequest* pruned_requests,
                                           std::uint32_t& request_count,
                                           const std::uint32_t* request_keys,
                                           TerminalPruneWorkspace workspace,
                                           cudaStream_t stream) {
    if (request_count == 0u)
        return cudaSuccess;

    const std::uint32_t blocks = block_count(request_count, kKernelBlockSize);
    fill_terminal_cell_end_kernel<<<blocks, kKernelBlockSize, 0, stream>>>(
        requests, request_count, workspace.coverage_offsets);
    cudaError_t status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    status = algo::cuda::scan::exclusive_scan<
        algo::cuda::scan::Max<std::uint32_t>>(
        workspace.coverage_offsets, request_count, workspace.scan_workspace,
        workspace.scan_workspace_size, stream);
    if (status != cudaSuccess)
        return status;

    mark_pruned_terminal_requests_kernel<<<blocks, kKernelBlockSize, 0,
                                           stream>>>(
        requests, request_keys, workspace.coverage_offsets,
        request_count, workspace.emit_offsets);
    status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    status = algo::cuda::scan::inclusive_sum(
        workspace.emit_offsets, request_count, workspace.scan_workspace,
        workspace.scan_workspace_size, stream);
    if (status != cudaSuccess)
        return status;

    std::uint32_t pruned_count = 0u;
    status = cudaMemcpyAsync(&pruned_count,
                             workspace.emit_offsets + request_count - 1u,
                             sizeof(std::uint32_t),
                             cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess)
        return status;
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
        return status;

    compact_pruned_terminal_requests_kernel<<<blocks, kKernelBlockSize, 0,
                                              stream>>>(
        requests, request_keys, workspace.coverage_offsets,
        workspace.emit_offsets, request_count, pruned_requests);
    status = last_launch_status();
    if (status != cudaSuccess)
        return status;

    request_count = pruned_count;
    return cudaStreamSynchronize(stream);
}

} // namespace algo::svt::cuda::detail
