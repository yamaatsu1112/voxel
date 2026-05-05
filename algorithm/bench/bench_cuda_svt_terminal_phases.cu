#include <algo/svt/cuda.cuh>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace {

using algo::svt::cuda::TerminalLeafInput;
using algo::svt::cuda::TerminalNodeInput;
using BenchmarkGpuSvo = algo::svt::cuda::GpuSvo<(1u << 24), (1u << 24)>;

inline constexpr std::uint32_t kSphereRadius = 128u;
inline constexpr std::int32_t kAlignedSphereCenter =
    static_cast<std::int32_t>(algo::svt::cuda::kWorldVoxelCount / 2u);

enum class TerminalPhase {
    Count,
    Emit,
    Sort,
    Prune,
    Allocate,
    Apply,
    Release,
    Collapse
};

enum class TerminalScene { Empty, HalfOverlap };

struct Sphere {
    std::int32_t center_x;
    std::int32_t center_y;
    std::int32_t center_z;
};

struct TerminalSphereEdits {
    std::vector<TerminalNodeInput> nodes;
    std::vector<TerminalLeafInput> leaves;
    std::uint64_t covered_voxels = 0u;
    std::uint32_t request_capacity = 0u;
};

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::uint64_t square(std::int64_t value) {
    return static_cast<std::uint64_t>(value * value);
}

bool voxel_inside_sphere(Sphere sphere, std::int32_t x, std::int32_t y,
                         std::int32_t z) {
    const std::int64_t dx = static_cast<std::int64_t>(x) - sphere.center_x;
    const std::int64_t dy = static_cast<std::int64_t>(y) - sphere.center_y;
    const std::int64_t dz = static_cast<std::int64_t>(z) - sphere.center_z;
    return square(dx) + square(dy) + square(dz) <=
           square(static_cast<std::int64_t>(kSphereRadius));
}

bool cell_inside_sphere(Sphere sphere, std::int32_t x0, std::int32_t y0,
                        std::int32_t z0, std::int32_t size) {
    const std::int32_t x1 = x0 + size - 1;
    const std::int32_t y1 = y0 + size - 1;
    const std::int32_t z1 = z0 + size - 1;
    for (const std::int32_t z : {z0, z1})
        for (const std::int32_t y : {y0, y1})
            for (const std::int32_t x : {x0, x1})
                if (!voxel_inside_sphere(sphere, x, y, z))
                    return false;
    return true;
}

bool cell_intersects_sphere(Sphere sphere, std::int32_t x0, std::int32_t y0,
                            std::int32_t z0, std::int32_t size) {
    const std::int32_t x1 = x0 + size - 1;
    const std::int32_t y1 = y0 + size - 1;
    const std::int32_t z1 = z0 + size - 1;
    const std::int64_t closest_x = std::clamp(sphere.center_x, x0, x1);
    const std::int64_t closest_y = std::clamp(sphere.center_y, y0, y1);
    const std::int64_t closest_z = std::clamp(sphere.center_z, z0, z1);
    const std::int64_t dx = closest_x - sphere.center_x;
    const std::int64_t dy = closest_y - sphere.center_y;
    const std::int64_t dz = closest_z - sphere.center_z;
    return square(dx) + square(dy) + square(dz) <=
           square(static_cast<std::int64_t>(kSphereRadius));
}

std::uint32_t child_prefix(std::uint32_t prefix, std::uint32_t child) {
    return (prefix << algo::svt::cuda::kGroupSizeExp) | child;
}

std::uint64_t full_cell_voxel_count(std::uint32_t depth) {
    const std::uint32_t log_size = algo::svt::cuda::kLeafVoxelCountExp +
                                   (algo::svt::cuda::kMaxDepth - depth) *
                                       algo::svt::cuda::kBranchFactorExp;
    const std::uint64_t size = 1ull << log_size;
    return size * size * size;
}

std::uint64_t append_terminal_leaf(Sphere sphere, TerminalSphereEdits& edits,
                                   std::uint32_t leaf_x, std::uint32_t leaf_y,
                                   std::uint32_t leaf_z) {
    std::uint64_t mask = 0u;
    std::uint64_t covered = 0u;
    const std::uint32_t base_x = leaf_x << algo::svt::cuda::kLeafVoxelCountExp;
    const std::uint32_t base_y = leaf_y << algo::svt::cuda::kLeafVoxelCountExp;
    const std::uint32_t base_z = leaf_z << algo::svt::cuda::kLeafVoxelCountExp;

    for (std::uint32_t z = 0u; z < algo::svt::cuda::kLeafVoxelCount; ++z)
        for (std::uint32_t y = 0u; y < algo::svt::cuda::kLeafVoxelCount; ++y)
            for (std::uint32_t x = 0u; x < algo::svt::cuda::kLeafVoxelCount;
                 ++x) {
                if (!voxel_inside_sphere(
                        sphere, static_cast<std::int32_t>(base_x + x),
                        static_cast<std::int32_t>(base_y + y),
                        static_cast<std::int32_t>(base_z + z))) {
                    continue;
                }
                const std::uint32_t bit =
                    x | (y << algo::svt::cuda::kLeafVoxelCountExp) |
                    (z << (algo::svt::cuda::kLeafVoxelCountExp * 2u));
                mask |= 1ull << bit;
                ++covered;
            }

    if (mask != 0u) {
        edits.leaves.push_back(TerminalLeafInput{
            algo::svt::cuda::make_leaf_key(leaf_x, leaf_y, leaf_z), mask, 0, 0,
            0});
        edits.request_capacity += static_cast<std::uint32_t>(covered);
    }
    return covered;
}

void append_terminal_cell(Sphere sphere, TerminalSphereEdits& edits,
                          std::uint32_t depth, std::uint32_t prefix,
                          std::uint32_t leaf_x, std::uint32_t leaf_y,
                          std::uint32_t leaf_z) {
    const std::uint32_t leaf_span = 1u << (algo::svt::cuda::kMaxDepth - depth);
    const std::int32_t voxel_x = static_cast<std::int32_t>(
        leaf_x << algo::svt::cuda::kLeafVoxelCountExp);
    const std::int32_t voxel_y = static_cast<std::int32_t>(
        leaf_y << algo::svt::cuda::kLeafVoxelCountExp);
    const std::int32_t voxel_z = static_cast<std::int32_t>(
        leaf_z << algo::svt::cuda::kLeafVoxelCountExp);
    const std::int32_t voxel_size = static_cast<std::int32_t>(
        leaf_span << algo::svt::cuda::kLeafVoxelCountExp);

    if (!cell_intersects_sphere(sphere, voxel_x, voxel_y, voxel_z, voxel_size))
        return;

    if (cell_inside_sphere(sphere, voxel_x, voxel_y, voxel_z, voxel_size)) {
        edits.nodes.push_back(TerminalNodeInput{depth, prefix, 0, 0, 0});
        edits.covered_voxels += full_cell_voxel_count(depth);
        ++edits.request_capacity;
        return;
    }

    if (depth == algo::svt::cuda::kMaxDepth) {
        edits.covered_voxels +=
            append_terminal_leaf(sphere, edits, leaf_x, leaf_y, leaf_z);
        return;
    }

    const std::uint32_t child_leaf_span = leaf_span >> 1u;
    for (std::uint32_t child = 0u; child < algo::svt::cuda::kGroupSize;
         ++child) {
        append_terminal_cell(
            sphere, edits, depth + 1u, child_prefix(prefix, child),
            leaf_x + ((child & 1u) != 0u ? child_leaf_span : 0u),
            leaf_y + ((child & 2u) != 0u ? child_leaf_span : 0u),
            leaf_z + ((child & 4u) != 0u ? child_leaf_span : 0u));
    }
}

TerminalSphereEdits make_terminal_sphere_edits(Sphere sphere) {
    TerminalSphereEdits edits;
    append_terminal_cell(sphere, edits, 0u, 0u, 0u, 0u, 0u);
    return edits;
}

Sphere centered_sphere() {
    return {kAlignedSphereCenter, kAlignedSphereCenter, kAlignedSphereCenter};
}

Sphere half_overlap_sphere() {
    return {kAlignedSphereCenter +
                static_cast<std::int32_t>(kSphereRadius / 2u),
            kAlignedSphereCenter, kAlignedSphereCenter};
}

template <class T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    bool allocate(std::size_t count, benchmark::State& state,
                  const char* label) {
        const std::size_t allocated_count = count == 0 ? 1 : count;
        const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&ptr_),
                                              sizeof(T) * allocated_count);
        if (status != cudaSuccess) {
            state.SkipWithError(label);
            return false;
        }
        return true;
    }

    [[nodiscard]] T* get() const { return ptr_; }

  private:
    T* ptr_ = nullptr;
};

template <class T>
bool copy_to_device(T* dst, const std::vector<T>& src, benchmark::State& state,
                    const char* label) {
    if (src.empty())
        return true;
    const cudaError_t status = cudaMemcpy(
        dst, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        state.SkipWithError(label);
        return false;
    }
    return true;
}

template <class T>
cudaError_t copy_device_to_device(T* dst, const T* src, std::size_t count) {
    if (count == 0)
        return cudaSuccess;
    return cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToDevice);
}

bool create_events(cudaEvent_t* start, cudaEvent_t* stop,
                   benchmark::State& state) {
    if (cudaEventCreate(start) != cudaSuccess ||
        cudaEventCreate(stop) != cudaSuccess) {
        if (*start != nullptr)
            cudaEventDestroy(*start);
        if (*stop != nullptr)
            cudaEventDestroy(*stop);
        state.SkipWithError("cudaEventCreate failed");
        return false;
    }
    return true;
}

bool record_start(cudaEvent_t start, benchmark::State& state) {
    if (cudaEventRecord(start) != cudaSuccess) {
        state.SkipWithError("cudaEventRecord(start) failed");
        return false;
    }
    return true;
}

bool record_elapsed(cudaEvent_t start, cudaEvent_t stop,
                    benchmark::State& state) {
    if (cudaEventRecord(stop) != cudaSuccess) {
        state.SkipWithError("cudaEventRecord(stop) failed");
        return false;
    }
    if (cudaEventSynchronize(stop) != cudaSuccess) {
        state.SkipWithError("cudaEventSynchronize(stop) failed");
        return false;
    }
    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) {
        state.SkipWithError("cudaEventElapsedTime failed");
        return false;
    }
    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
    return true;
}

void destroy_events(cudaEvent_t start, cudaEvent_t stop) {
    if (stop != nullptr)
        cudaEventDestroy(stop);
    if (start != nullptr)
        cudaEventDestroy(start);
}

void set_terminal_counters(benchmark::State& state,
                           const TerminalSphereEdits& edits) {
    const auto node_count = static_cast<std::uint32_t>(edits.nodes.size());
    const auto leaf_count = static_cast<std::uint32_t>(edits.leaves.size());
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(edits.covered_voxels));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(node_count * sizeof(TerminalNodeInput) +
                             leaf_count * sizeof(TerminalLeafInput)));
    state.counters["voxels"] = static_cast<double>(edits.covered_voxels);
    state.counters["terminal_nodes"] = static_cast<double>(node_count);
    state.counters["terminal_leaves"] = static_cast<double>(leaf_count);
    state.counters["terminal_inputs"] =
        static_cast<double>(node_count + leaf_count);
    state.counters["terminal_requests"] =
        static_cast<double>(edits.request_capacity);
}

cudaError_t clear_terminal_status(
    algo::svt::cuda::detail::TerminalEditWorkspace& workspace) {
    cudaError_t status =
        cudaMemsetAsync(workspace.status, 0, sizeof(cudaError_t));
    if (status != cudaSuccess)
        return status;
    return cudaMemsetAsync(workspace.detached_child_count, 0,
                           sizeof(std::uint32_t));
}

cudaError_t count_terminal_requests(
    const TerminalNodeInput* nodes, std::uint32_t node_count,
    const TerminalLeafInput* leaves, std::uint32_t leaf_count,
    algo::svt::cuda::detail::TerminalEditWorkspace& workspace,
    std::uint32_t& request_count) {
    namespace cuda_detail = algo::svt::cuda::detail;

    const std::uint32_t count_segment_count = node_count * 2u + leaf_count;
    const std::uint32_t count_segment_blocks = cuda_detail::block_count(
        count_segment_count, cuda_detail::kKernelBlockSize);
    cuda_detail::count_terminal_requests_kernel<<<
        count_segment_blocks, cuda_detail::kKernelBlockSize>>>(
        nodes, node_count, leaves, leaf_count, workspace.request_offsets);
    cudaError_t status = cuda_detail::last_launch_status();
    if (status != cudaSuccess)
        return status;

    auto count_sort_workspace = cuda_detail::create_terminal_count_sort_workspace(
        workspace.phase_scratch, count_segment_count, workspace.request_capacity);
    status = algo::cuda::scan::inclusive_sum(
        workspace.request_offsets, count_segment_count,
        count_sort_workspace.scan_workspace,
        count_sort_workspace.scan_workspace_size);
    if (status != cudaSuccess)
        return status;

    status = cudaMemcpyAsync(
        &request_count, workspace.request_offsets + count_segment_count - 1u,
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        return status;
    status = cudaStreamSynchronize(nullptr);
    if (status != cudaSuccess)
        return status;
    return request_count > workspace.request_capacity ? cudaErrorInvalidValue
                                                      : cudaSuccess;
}

cudaError_t emit_terminal_requests(
    const TerminalNodeInput* nodes, std::uint32_t node_count,
    const TerminalLeafInput* leaves, std::uint32_t leaf_count,
    algo::svt::cuda::detail::TerminalEditWorkspace& workspace,
    std::uint32_t request_count) {
    namespace cuda_detail = algo::svt::cuda::detail;

    const std::uint32_t count_segment_count = node_count * 2u + leaf_count;
    const std::uint32_t blocks =
        cuda_detail::block_count(request_count, cuda_detail::kKernelBlockSize);
    cuda_detail::emit_terminal_requests_kernel<<<
        blocks, cuda_detail::kKernelBlockSize>>>(
        nodes, node_count, leaves, leaf_count, workspace.request_offsets,
        count_segment_count, request_count, workspace.requests,
        workspace.request_keys);
    return cuda_detail::last_launch_status();
}

cudaError_t prepare_terminal_requests(
    const TerminalNodeInput* nodes, std::uint32_t node_count,
    const TerminalLeafInput* leaves, std::uint32_t leaf_count,
    algo::svt::cuda::detail::TerminalEditWorkspace& workspace,
    std::uint32_t& request_count, bool sorted, bool pruned) {
    namespace cuda_detail = algo::svt::cuda::detail;

    cudaError_t status = count_terminal_requests(
        nodes, node_count, leaves, leaf_count, workspace, request_count);
    if (status != cudaSuccess || request_count == 0u)
        return status;
    status = emit_terminal_requests(nodes, node_count, leaves, leaf_count,
                                    workspace, request_count);
    if (status != cudaSuccess)
        return status;
    if (!sorted)
        return cudaDeviceSynchronize();
    auto count_sort_workspace = cuda_detail::create_terminal_count_sort_workspace(
        workspace.phase_scratch, node_count * 2u + leaf_count,
        workspace.request_capacity);
    status = algo::cuda::sort::sort_pairs(
        workspace.request_keys, workspace.requests, request_count,
        count_sort_workspace.sort_workspace,
        count_sort_workspace.sort_workspace_size, nullptr);
    if (status != cudaSuccess)
        return status;
    if (!pruned)
        return cudaDeviceSynchronize();
    return cuda_detail::prune_terminal_requests(
        workspace.requests, workspace.pruned_requests, request_count,
        workspace.request_keys,
        cuda_detail::create_terminal_prune_workspace(workspace.phase_scratch,
                                                     workspace.request_capacity),
        nullptr);
}

cudaError_t
apply_terminal_writes(algo::svt::cuda::DeviceGpuSvo svo,
                      const algo::svt::cuda::TerminalRequest* requests,
                      algo::svt::cuda::detail::TerminalEditWorkspace& workspace,
                      std::uint32_t request_count) {
    namespace cuda_detail = algo::svt::cuda::detail;

    const std::uint32_t blocks =
        cuda_detail::block_count(request_count, cuda_detail::kKernelBlockSize);
    cuda_detail::apply_terminal_brick_requests_kernel<<<
        blocks, cuda_detail::kKernelBlockSize>>>(
        svo, requests, request_count, true, workspace.status);
    cudaError_t status = cuda_detail::last_launch_status();
    if (status != cudaSuccess)
        return status;

    cuda_detail::apply_terminal_cell_requests_kernel<<<
        blocks, cuda_detail::kKernelBlockSize>>>(
        svo, requests, request_count, true, workspace.status,
        workspace.detached_children, workspace.detached_child_count,
        workspace.request_capacity);
    status = cuda_detail::last_launch_status();
    if (status != cudaSuccess)
        return status;

    cudaError_t host_status = cudaSuccess;
    status = cudaMemcpyAsync(&host_status, workspace.status,
                             sizeof(cudaError_t), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        return status;
    status = cudaStreamSynchronize(nullptr);
    if (status != cudaSuccess)
        return status;
    return host_status;
}

template <class Release>
cudaError_t release_terminal_detached_children(
    algo::svt::cuda::DeviceGpuSvo svo,
    algo::svt::cuda::detail::TerminalEditWorkspace& workspace,
    std::uint32_t request_count) {
    namespace cuda_detail = algo::svt::cuda::detail;

    (void)request_count;
    cudaError_t status = cuda_detail::release_terminal_detached_children<
        Release>(
        svo, workspace.status,
        cuda_detail::terminal_release_impl<Release>::create_phase_workspace(
            workspace.phase_scratch, workspace.detached_children,
            workspace.detached_child_count, workspace.request_capacity),
        nullptr);
    if (status != cudaSuccess)
        return status;
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
        return status;

    cudaError_t host_status = cudaSuccess;
    status = cudaMemcpy(&host_status, workspace.status, sizeof(cudaError_t),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        return status;
    return host_status;
}

template <class Allocation = algo::svt::cuda::TerminalPlainDepthwiseAllocation,
          class Release = algo::svt::cuda::TerminalDepthwiseRelease>
void BM_CudaSvtTerminalPhase(benchmark::State& state, TerminalPhase phase,
                             TerminalScene scene = TerminalScene::Empty) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const TerminalSphereEdits edits = make_terminal_sphere_edits(
        scene == TerminalScene::HalfOverlap ? half_overlap_sphere()
                                            : centered_sphere());
    const std::optional<TerminalSphereEdits> preseed_edits =
        scene == TerminalScene::HalfOverlap
            ? std::optional<TerminalSphereEdits>(
                  make_terminal_sphere_edits(centered_sphere()))
            : std::nullopt;
    const auto node_count = static_cast<std::uint32_t>(edits.nodes.size());
    const auto leaf_count = static_cast<std::uint32_t>(edits.leaves.size());
    const auto preseed_node_count = static_cast<std::uint32_t>(
        preseed_edits ? preseed_edits->nodes.size() : 0u);
    const auto preseed_leaf_count = static_cast<std::uint32_t>(
        preseed_edits ? preseed_edits->leaves.size() : 0u);

    DeviceBuffer<TerminalNodeInput> d_nodes;
    DeviceBuffer<TerminalLeafInput> d_leaves;
    DeviceBuffer<TerminalNodeInput> d_preseed_nodes;
    DeviceBuffer<TerminalLeafInput> d_preseed_leaves;
    DeviceBuffer<std::byte> workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t workspace_size = std::max(
        algo::svt::cuda::detail::apply_terminal_edits_workspace_size<
            Allocation, Release>(
            node_count, leaf_count, edits.request_capacity),
        preseed_edits
            ? algo::svt::cuda::detail::apply_terminal_edits_workspace_size<
                  Allocation, Release>(
                  preseed_node_count, preseed_leaf_count,
                  preseed_edits->request_capacity)
            : std::size_t{0u});
    if (!d_nodes.allocate(node_count, state,
                          "cudaMalloc failed for terminal nodes") ||
        !d_leaves.allocate(leaf_count, state,
                           "cudaMalloc failed for terminal leaves") ||
        (preseed_edits &&
         (!d_preseed_nodes.allocate(preseed_node_count, state,
                                    "cudaMalloc failed for preseed nodes") ||
          !d_preseed_leaves.allocate(
              preseed_leaf_count, state,
              "cudaMalloc failed for preseed leaves"))) ||
        !workspace.allocate(workspace_size, state,
                            "cudaMalloc failed for workspace") ||
        !copy_to_device(d_nodes.get(), edits.nodes, state,
                        "cudaMemcpy failed for terminal nodes") ||
        !copy_to_device(d_leaves.get(), edits.leaves, state,
                        "cudaMemcpy failed for terminal leaves") ||
        (preseed_edits &&
         (!copy_to_device(d_preseed_nodes.get(), preseed_edits->nodes, state,
                          "cudaMemcpy failed for preseed nodes") ||
          !copy_to_device(d_preseed_leaves.get(), preseed_edits->leaves, state,
                          "cudaMemcpy failed for preseed leaves"))) ||
        !create_events(&start, &stop, state)) {
        destroy_events(start, stop);
        return;
    }

    algo::svt::cuda::detail::TerminalEditWorkspace typed_workspace =
        algo::svt::cuda::detail::create_terminal_edit_workspace<
            Allocation, Release>(
            workspace.get(), node_count, leaf_count, workspace_size);

    const bool needs_svo =
        phase == TerminalPhase::Allocate || phase == TerminalPhase::Apply ||
        phase == TerminalPhase::Release || phase == TerminalPhase::Collapse;
    std::optional<BenchmarkGpuSvo> svo;
    if (needs_svo) {
        svo.emplace();
        if (svo->status() != cudaSuccess) {
            state.SkipWithError("GpuSvo allocation failed");
            destroy_events(start, stop);
            return;
        }
    }

    DeviceBuffer<algo::svt::cuda::TerminalRequest> saved_requests;
    DeviceBuffer<std::uint32_t> saved_request_keys;
    std::uint32_t prepared_request_count = 0u;

    if (phase == TerminalPhase::Emit) {
        const cudaError_t status = count_terminal_requests(
            d_nodes.get(), node_count, d_leaves.get(), leaf_count,
            typed_workspace, prepared_request_count);
        if (status != cudaSuccess) {
            state.SkipWithError("terminal emit setup failed");
            destroy_events(start, stop);
            return;
        }
    } else if (phase == TerminalPhase::Sort) {
        const cudaError_t status = prepare_terminal_requests(
            d_nodes.get(), node_count, d_leaves.get(), leaf_count,
            typed_workspace, prepared_request_count, false, false);
        if (status != cudaSuccess ||
            !saved_requests.allocate(prepared_request_count, state,
                                     "cudaMalloc failed for saved requests") ||
            !saved_request_keys.allocate(prepared_request_count, state,
                                         "cudaMalloc failed for saved keys") ||
            copy_device_to_device(saved_requests.get(),
                                  typed_workspace.requests,
                                  prepared_request_count) != cudaSuccess ||
            copy_device_to_device(saved_request_keys.get(),
                                  typed_workspace.request_keys,
                                  prepared_request_count) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            state.SkipWithError("terminal sort setup failed");
            destroy_events(start, stop);
            return;
        }
    } else if (phase == TerminalPhase::Prune) {
        const cudaError_t status = prepare_terminal_requests(
            d_nodes.get(), node_count, d_leaves.get(), leaf_count,
            typed_workspace, prepared_request_count, true, false);
        if (status != cudaSuccess ||
            !saved_requests.allocate(prepared_request_count, state,
                                     "cudaMalloc failed for saved requests") ||
            !saved_request_keys.allocate(prepared_request_count, state,
                                         "cudaMalloc failed for saved keys") ||
            copy_device_to_device(saved_requests.get(),
                                  typed_workspace.requests,
                                  prepared_request_count) != cudaSuccess ||
            copy_device_to_device(saved_request_keys.get(),
                                  typed_workspace.request_keys,
                                  prepared_request_count) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            state.SkipWithError("terminal prune setup failed");
            destroy_events(start, stop);
            return;
        }
    } else if (phase == TerminalPhase::Allocate ||
               phase == TerminalPhase::Apply ||
               phase == TerminalPhase::Release ||
               phase == TerminalPhase::Collapse) {
        const cudaError_t status = prepare_terminal_requests(
            d_nodes.get(), node_count, d_leaves.get(), leaf_count,
            typed_workspace, prepared_request_count, true, true);
        if (status != cudaSuccess ||
            (preseed_edits &&
             (!saved_requests.allocate(
                  prepared_request_count, state,
                  "cudaMalloc failed for saved requests") ||
              copy_device_to_device(saved_requests.get(),
                                    typed_workspace.pruned_requests,
                                    prepared_request_count) != cudaSuccess ||
              cudaDeviceSynchronize() != cudaSuccess))) {
            state.SkipWithError("terminal request setup failed");
            destroy_events(start, stop);
            return;
        }
    }

    for (auto _ : state) {
        std::uint32_t request_count = prepared_request_count;
        cudaError_t status = cudaSuccess;
        if (phase == TerminalPhase::Count) {
            if (!record_start(start, state))
                break;
            status = count_terminal_requests(d_nodes.get(), node_count,
                                             d_leaves.get(), leaf_count,
                                             typed_workspace, request_count);
        } else if (phase == TerminalPhase::Emit) {
            if (!record_start(start, state))
                break;
            status = emit_terminal_requests(d_nodes.get(), node_count,
                                            d_leaves.get(), leaf_count,
                                            typed_workspace, request_count);
        } else if (phase == TerminalPhase::Sort) {
            status = copy_device_to_device(typed_workspace.requests,
                                           saved_requests.get(), request_count);
            if (status == cudaSuccess) {
                status = copy_device_to_device(typed_workspace.request_keys,
                                               saved_request_keys.get(),
                                               request_count);
            }
            if (status == cudaSuccess && !record_start(start, state))
                break;
            if (status == cudaSuccess) {
                auto count_sort_workspace =
                    algo::svt::cuda::detail::
                        create_terminal_count_sort_workspace(
                            typed_workspace.phase_scratch,
                            node_count * 2u + leaf_count,
                            typed_workspace.request_capacity);
                status = algo::cuda::sort::sort_pairs(
                    typed_workspace.request_keys, typed_workspace.requests,
                    request_count, count_sort_workspace.sort_workspace,
                    count_sort_workspace.sort_workspace_size, nullptr);
            }
        } else if (phase == TerminalPhase::Prune) {
            status = copy_device_to_device(typed_workspace.requests,
                                           saved_requests.get(), request_count);
            if (status == cudaSuccess) {
                status = copy_device_to_device(typed_workspace.request_keys,
                                               saved_request_keys.get(),
                                               request_count);
            }
            if (status == cudaSuccess && !record_start(start, state))
                break;
            if (status == cudaSuccess) {
                status = algo::svt::cuda::detail::prune_terminal_requests(
                    typed_workspace.requests, typed_workspace.pruned_requests,
                    request_count, typed_workspace.request_keys,
                    algo::svt::cuda::detail::create_terminal_prune_workspace(
                        typed_workspace.phase_scratch,
                        typed_workspace.request_capacity),
                    nullptr);
            }
        } else {
            const algo::svt::cuda::TerminalRequest* active_requests =
                typed_workspace.pruned_requests;
            if (algo::svt::cuda::reset_svo(svo->view()) != cudaSuccess ||
                clear_terminal_status(typed_workspace) != cudaSuccess) {
                state.SkipWithError("terminal phase setup failed");
                break;
            }
            if (preseed_edits) {
                status = algo::svt::cuda::place_terminal_edits<
                    algo::svt::cuda::TerminalEditConfig<Allocation, Release>>(
                    svo->view(), d_preseed_nodes.get(), preseed_node_count,
                    d_preseed_leaves.get(), preseed_leaf_count, workspace.get(),
                    workspace_size);
                if (status == cudaSuccess) {
                    status =
                        copy_device_to_device(typed_workspace.pruned_requests,
                                              saved_requests.get(),
                                              request_count);
                }
                if (status == cudaSuccess)
                    status = clear_terminal_status(typed_workspace);
            }
            if (status == cudaSuccess && phase != TerminalPhase::Allocate) {
                status =
                    algo::svt::cuda::detail::
                        allocate_terminal_request_paths<Allocation>(
                            svo->view(), active_requests, request_count,
                            typed_workspace.request_keys,
                            typed_workspace.phase_scratch, nullptr);
            }
            if (status == cudaSuccess && (phase == TerminalPhase::Release ||
                                          phase == TerminalPhase::Collapse)) {
                status = apply_terminal_writes(svo->view(), active_requests,
                                               typed_workspace,
                                               request_count);
            }
            if (status == cudaSuccess && phase == TerminalPhase::Collapse) {
                status = release_terminal_detached_children<Release>(
                    svo->view(), typed_workspace, request_count);
            }
            if (status == cudaSuccess && record_start(start, state)) {
                if (phase == TerminalPhase::Allocate) {
                    status = algo::svt::cuda::detail::
                        allocate_terminal_request_paths<Allocation>(
                            svo->view(), active_requests, request_count,
                            typed_workspace.request_keys,
                            typed_workspace.phase_scratch, nullptr);
                } else if (phase == TerminalPhase::Apply) {
                    status = apply_terminal_writes(
                        svo->view(), active_requests, typed_workspace,
                        request_count);
                } else if (phase == TerminalPhase::Release) {
                    status = release_terminal_detached_children<Release>(
                        svo->view(), typed_workspace, request_count);
                } else {
                    status =
                        algo::svt::cuda::detail::collapse_terminal_requests(
                            svo->view(), active_requests, request_count,
                            algo::svt::cuda::detail::
                                create_terminal_collapse_workspace(
                                    typed_workspace.phase_scratch,
                                    typed_workspace.request_keys,
                                    typed_workspace.request_capacity),
                            nullptr);
                }
            }
        }

        if (status != cudaSuccess) {
            state.SkipWithError("terminal phase failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
        benchmark::DoNotOptimize(request_count);
    }

    set_terminal_counters(state, edits);
    if (preseed_edits) {
        state.counters["preseed_terminal_inputs"] =
            static_cast<double>(preseed_node_count + preseed_leaf_count);
        state.counters["preseed_voxels"] =
            static_cast<double>(preseed_edits->covered_voxels);
        state.counters["overlap_offset_voxels"] =
            static_cast<double>(kSphereRadius / 2u);
    }
    destroy_events(start, stop);
}

#define TERMINAL_PHASE_BENCHMARKS(X)                                           \
    X(Count, TerminalPhase::Count)                                             \
    X(Emit, TerminalPhase::Emit)                                               \
    X(Sort, TerminalPhase::Sort)                                               \
    X(Prune, TerminalPhase::Prune)                                             \
    X(AllocatePlainDepthwise, TerminalPhase::Allocate)                         \
    X(AllocateCompactAllDepth, TerminalPhase::Allocate, TerminalScene::Empty,  \
      algo::svt::cuda::TerminalCompactAllDepthAllocation)                      \
    X(Apply, TerminalPhase::Apply)                                             \
    X(Release, TerminalPhase::Release)                                         \
    X(ReleaseFrontier, TerminalPhase::Release, TerminalScene::Empty,           \
      algo::svt::cuda::TerminalPlainDepthwiseAllocation,                       \
      algo::svt::cuda::TerminalFrontierRelease)                                \
    X(Collapse, TerminalPhase::Collapse)                                       \
    X(AllocatePlainDepthwiseHalfOverlap, TerminalPhase::Allocate,              \
      TerminalScene::HalfOverlap)                                              \
    X(AllocateCompactAllDepthHalfOverlap, TerminalPhase::Allocate,             \
      TerminalScene::HalfOverlap,                                              \
      algo::svt::cuda::TerminalCompactAllDepthAllocation)                      \
    X(ApplyHalfOverlap, TerminalPhase::Apply, TerminalScene::HalfOverlap)      \
    X(ReleaseHalfOverlap, TerminalPhase::Release, TerminalScene::HalfOverlap)  \
    X(ReleaseFrontierHalfOverlap, TerminalPhase::Release,                      \
      TerminalScene::HalfOverlap,                                              \
      algo::svt::cuda::TerminalPlainDepthwiseAllocation,                       \
      algo::svt::cuda::TerminalFrontierRelease)                                \
    X(CollapseHalfOverlap, TerminalPhase::Collapse, TerminalScene::HalfOverlap)

#define DEFINE_TERMINAL_PHASE_BENCHMARK_2(Name, Phase)                         \
    void BM_CudaSvtTerminalPhase##Name(benchmark::State& state) {              \
        BM_CudaSvtTerminalPhase(state, Phase);                                 \
    }

#define DEFINE_TERMINAL_PHASE_BENCHMARK_3(Name, Phase, Scene)                  \
    void BM_CudaSvtTerminalPhase##Name(benchmark::State& state) {              \
        BM_CudaSvtTerminalPhase(state, Phase, Scene);                          \
    }

#define DEFINE_TERMINAL_PHASE_BENCHMARK_4(Name, Phase, Scene, Allocation)      \
    void BM_CudaSvtTerminalPhase##Name(benchmark::State& state) {              \
        BM_CudaSvtTerminalPhase<Allocation>(state, Phase, Scene);              \
    }

#define DEFINE_TERMINAL_PHASE_BENCHMARK_5(Name, Phase, Scene, Allocation,      \
                                          Release)                             \
    void BM_CudaSvtTerminalPhase##Name(benchmark::State& state) {              \
        BM_CudaSvtTerminalPhase<Allocation, Release>(state, Phase, Scene);     \
    }

#define DEFINE_TERMINAL_PHASE_BENCHMARK_SELECT(_1, _2, _3, _4, _5, NAME, ...)  \
    NAME
#define DEFINE_TERMINAL_PHASE_BENCHMARK(...)                                   \
    DEFINE_TERMINAL_PHASE_BENCHMARK_SELECT(                                    \
        __VA_ARGS__, DEFINE_TERMINAL_PHASE_BENCHMARK_5,                        \
        DEFINE_TERMINAL_PHASE_BENCHMARK_4,                                     \
        DEFINE_TERMINAL_PHASE_BENCHMARK_3,                                     \
        DEFINE_TERMINAL_PHASE_BENCHMARK_2)(__VA_ARGS__)

TERMINAL_PHASE_BENCHMARKS(DEFINE_TERMINAL_PHASE_BENCHMARK)

#undef DEFINE_TERMINAL_PHASE_BENCHMARK_SELECT
#undef DEFINE_TERMINAL_PHASE_BENCHMARK_5
#undef DEFINE_TERMINAL_PHASE_BENCHMARK_4
#undef DEFINE_TERMINAL_PHASE_BENCHMARK_3
#undef DEFINE_TERMINAL_PHASE_BENCHMARK_2
#undef DEFINE_TERMINAL_PHASE_BENCHMARK

#define REGISTER_TERMINAL_PHASE_BENCHMARK(Name, ...)                           \
    BENCHMARK(BM_CudaSvtTerminalPhase##Name)->UseManualTime();

TERMINAL_PHASE_BENCHMARKS(REGISTER_TERMINAL_PHASE_BENCHMARK)

#undef REGISTER_TERMINAL_PHASE_BENCHMARK
#undef TERMINAL_PHASE_BENCHMARKS

} // namespace
