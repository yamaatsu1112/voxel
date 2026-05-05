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
using algo::svt::cuda::VoxelEdit;
using BenchmarkGpuSvo = algo::svt::cuda::GpuSvo<(1u << 24), (1u << 24)>;

inline constexpr std::uint32_t kSphereRadius = 128u;
inline constexpr std::int32_t kAlignedSphereCenter =
    static_cast<std::int32_t>(algo::svt::cuda::kWorldVoxelCount / 2u);

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

TerminalSphereEdits with_world_offset(TerminalSphereEdits edits,
                                      std::int32_t offset_x,
                                      std::int32_t offset_y,
                                      std::int32_t offset_z) {
    for (TerminalNodeInput& node : edits.nodes) {
        node.worldOffsetX = offset_x;
        node.worldOffsetY = offset_y;
        node.worldOffsetZ = offset_z;
    }
    for (TerminalLeafInput& leaf : edits.leaves) {
        leaf.worldOffsetX = offset_x;
        leaf.worldOffsetY = offset_y;
        leaf.worldOffsetZ = offset_z;
    }
    return edits;
}

std::uint32_t terminal_workspace_request_capacity(
    const TerminalSphereEdits& edits) {
    return std::max(edits.request_capacity,
                    static_cast<std::uint32_t>(edits.covered_voxels));
}

std::vector<VoxelEdit> make_voxel_sphere_edits(Sphere sphere) {
    std::vector<VoxelEdit> edits;
    const std::int32_t begin_x =
        sphere.center_x - static_cast<std::int32_t>(kSphereRadius);
    const std::int32_t end_x =
        sphere.center_x + static_cast<std::int32_t>(kSphereRadius);
    const std::int32_t begin_y =
        sphere.center_y - static_cast<std::int32_t>(kSphereRadius);
    const std::int32_t end_y =
        sphere.center_y + static_cast<std::int32_t>(kSphereRadius);
    const std::int32_t begin_z =
        sphere.center_z - static_cast<std::int32_t>(kSphereRadius);
    const std::int32_t end_z =
        sphere.center_z + static_cast<std::int32_t>(kSphereRadius);

    for (std::int32_t z = begin_z; z <= end_z; ++z)
        for (std::int32_t y = begin_y; y <= end_y; ++y)
            for (std::int32_t x = begin_x; x <= end_x; ++x)
                if (voxel_inside_sphere(sphere, x, y, z))
                    edits.push_back(VoxelEdit{static_cast<std::uint32_t>(x),
                                              static_cast<std::uint32_t>(y),
                                              static_cast<std::uint32_t>(z)});
    return edits;
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

bool create_events(cudaEvent_t* start, cudaEvent_t* stop,
                   benchmark::State& state) {
    if (cudaEventCreate(start) != cudaSuccess ||
        cudaEventCreate(stop) != cudaSuccess) {
        if (*start != nullptr) {
            cudaEventDestroy(*start);
            *start = nullptr;
        }
        if (*stop != nullptr) {
            cudaEventDestroy(*stop);
            *stop = nullptr;
        }
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

void set_shape_counters(benchmark::State& state, std::uint64_t voxel_count) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(voxel_count));
    state.counters["voxels"] = static_cast<double>(voxel_count);
}

void BM_CudaSvtPlaceVoxelDefaultSphere(benchmark::State& state) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto edits = make_voxel_sphere_edits(
        {kAlignedSphereCenter, kAlignedSphereCenter, kAlignedSphereCenter});
    const auto count = static_cast<std::uint32_t>(edits.size());

    DeviceBuffer<VoxelEdit> d_edits;
    DeviceBuffer<std::byte> workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size(count);
    BenchmarkGpuSvo svo;
    if (svo.status() != cudaSuccess) {
        state.SkipWithError("GpuSvo allocation failed");
        return;
    }

    if (!d_edits.allocate(count, state, "cudaMalloc failed for voxel edits") ||
        !workspace.allocate(workspace_size, state,
                            "cudaMalloc failed for workspace") ||
        !copy_to_device(d_edits.get(), edits, state,
                        "cudaMemcpy failed for voxel edits") ||
        !create_events(&start, &stop, state)) {
        destroy_events(start, stop);
        return;
    }

    for (auto _ : state) {
        if (algo::svt::cuda::reset_svo(svo.view()) != cudaSuccess) {
            state.SkipWithError("reset_svo failed");
            break;
        }
        if (!record_start(start, state))
            break;
        if (algo::svt::cuda::place_voxel_edits(svo.view(), d_edits.get(), count,
                                               workspace.get(),
                                               workspace_size) != cudaSuccess) {
            state.SkipWithError("place_voxel_edits failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
    }

    set_shape_counters(state, edits.size());
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(edits.size()) *
                            static_cast<int64_t>(sizeof(VoxelEdit)));
    destroy_events(start, stop);
}

void BM_CudaSvtPlaceTerminalSphere(benchmark::State& state,
                                   std::int32_t offset_x, std::int32_t offset_y,
                                   std::int32_t offset_z,
                                   bool half_overlap = false) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const TerminalSphereEdits edits = with_world_offset(
        make_terminal_sphere_edits(half_overlap ? half_overlap_sphere()
                                                : centered_sphere()),
        offset_x, offset_y, offset_z);
    const std::optional<TerminalSphereEdits> preseed_edits =
        half_overlap ? std::optional<TerminalSphereEdits>(
                           make_terminal_sphere_edits(centered_sphere()))
                     : std::nullopt;
    const auto node_count = static_cast<std::uint32_t>(edits.nodes.size());
    const auto leaf_count = static_cast<std::uint32_t>(edits.leaves.size());
    const auto input_count =
        static_cast<std::uint32_t>(node_count + leaf_count);
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
            algo::svt::cuda::DefaultTerminalEditConfig::allocation,
            algo::svt::cuda::DefaultTerminalEditConfig::release>(
            node_count, leaf_count, terminal_workspace_request_capacity(edits)),
        preseed_edits
            ? algo::svt::cuda::detail::apply_terminal_edits_workspace_size<
                  algo::svt::cuda::DefaultTerminalEditConfig::allocation,
                  algo::svt::cuda::DefaultTerminalEditConfig::release>(
                  preseed_node_count, preseed_leaf_count,
                  terminal_workspace_request_capacity(*preseed_edits))
            : std::size_t{0u});
    BenchmarkGpuSvo svo;
    if (svo.status() != cudaSuccess) {
        state.SkipWithError("GpuSvo allocation failed");
        return;
    }

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

    for (auto _ : state) {
        if (algo::svt::cuda::reset_svo(svo.view()) != cudaSuccess) {
            state.SkipWithError("reset_svo failed");
            break;
        }
        if (preseed_edits &&
            algo::svt::cuda::place_terminal_edits(
                svo.view(), d_preseed_nodes.get(), preseed_node_count,
                d_preseed_leaves.get(), preseed_leaf_count, workspace.get(),
                workspace_size) != cudaSuccess) {
            state.SkipWithError("preseed place_terminal_edits failed");
            break;
        }
        if (!record_start(start, state))
            break;
        if (algo::svt::cuda::place_terminal_edits(
                svo.view(), d_nodes.get(), node_count, d_leaves.get(),
                leaf_count, workspace.get(), workspace_size) != cudaSuccess) {
            state.SkipWithError("place_terminal_edits failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
    }

    set_shape_counters(state, edits.covered_voxels);
    state.counters["terminal_nodes"] = static_cast<double>(node_count);
    state.counters["terminal_leaves"] = static_cast<double>(leaf_count);
    state.counters["terminal_inputs"] = static_cast<double>(input_count);
    state.counters["terminal_requests"] =
        static_cast<double>(edits.request_capacity);
    if (preseed_edits) {
        state.counters["preseed_terminal_inputs"] =
            static_cast<double>(preseed_node_count + preseed_leaf_count);
        state.counters["preseed_voxels"] =
            static_cast<double>(preseed_edits->covered_voxels);
        state.counters["overlap_offset_voxels"] =
            static_cast<double>(kSphereRadius / 2u);
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(node_count * sizeof(TerminalNodeInput) +
                             leaf_count * sizeof(TerminalLeafInput)));
    destroy_events(start, stop);
}

void BM_CudaSvtPlaceTerminalSphereAligned(benchmark::State& state) {
    BM_CudaSvtPlaceTerminalSphere(state, 0, 0, 0);
}

void BM_CudaSvtPlaceTerminalSphereUnaligned(benchmark::State& state) {
    BM_CudaSvtPlaceTerminalSphere(state, 1, 1, 1);
}

void BM_CudaSvtPlaceTerminalSphereHalfOverlap(benchmark::State& state) {
    BM_CudaSvtPlaceTerminalSphere(state, 0, 0, 0, true);
}

BENCHMARK(BM_CudaSvtPlaceVoxelDefaultSphere)->UseManualTime();
BENCHMARK(BM_CudaSvtPlaceTerminalSphereAligned)->UseManualTime();
BENCHMARK(BM_CudaSvtPlaceTerminalSphereUnaligned)->UseManualTime();
BENCHMARK(BM_CudaSvtPlaceTerminalSphereHalfOverlap)->UseManualTime();

} // namespace
