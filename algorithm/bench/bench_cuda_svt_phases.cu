#include <algo/svt/cuda.cuh>
#include <algo/svt/cuda/voxel/detail/dispatch_capacity.cuh>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using algo::svt::cuda::VoxelEdit;
using EditGenerator = std::vector<VoxelEdit> (*)(std::uint32_t);
using BenchmarkGpuSvo = algo::svt::cuda::GpuSvo<(1u << 24), (1u << 24)>;

inline constexpr std::int32_t kAlignedSphereCenter =
    static_cast<std::int32_t>(algo::svt::cuda::kWorldVoxelCount / 2u);

struct Sphere {
    std::int32_t center_x;
    std::int32_t center_y;
    std::int32_t center_z;
};

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::uint64_t square(std::int64_t value) {
    return static_cast<std::uint64_t>(value * value);
}

bool voxel_inside_sphere(Sphere sphere, std::int32_t x, std::int32_t y,
                         std::int32_t z, std::uint32_t radius) {
    const std::int64_t dx = static_cast<std::int64_t>(x) - sphere.center_x;
    const std::int64_t dy = static_cast<std::int64_t>(y) - sphere.center_y;
    const std::int64_t dz = static_cast<std::int64_t>(z) - sphere.center_z;
    return square(dx) + square(dy) + square(dz) <=
           square(static_cast<std::int64_t>(radius));
}

Sphere centered_sphere() {
    return {kAlignedSphereCenter, kAlignedSphereCenter, kAlignedSphereCenter};
}

Sphere half_overlap_sphere(std::uint32_t radius) {
    return {kAlignedSphereCenter +
                static_cast<std::int32_t>(radius / 2u),
            kAlignedSphereCenter, kAlignedSphereCenter};
}

std::vector<VoxelEdit> make_voxel_sphere_edits(Sphere sphere,
                                               std::uint32_t radius) {
    std::vector<VoxelEdit> edits;
    const std::int32_t begin_x =
        sphere.center_x - static_cast<std::int32_t>(radius);
    const std::int32_t end_x =
        sphere.center_x + static_cast<std::int32_t>(radius);
    const std::int32_t begin_y =
        sphere.center_y - static_cast<std::int32_t>(radius);
    const std::int32_t end_y =
        sphere.center_y + static_cast<std::int32_t>(radius);
    const std::int32_t begin_z =
        sphere.center_z - static_cast<std::int32_t>(radius);
    const std::int32_t end_z =
        sphere.center_z + static_cast<std::int32_t>(radius);

    for (std::int32_t z = begin_z; z <= end_z; ++z)
        for (std::int32_t y = begin_y; y <= end_y; ++y)
            for (std::int32_t x = begin_x; x <= end_x; ++x)
                if (voxel_inside_sphere(sphere, x, y, z, radius))
                    edits.push_back(VoxelEdit{static_cast<std::uint32_t>(x),
                                              static_cast<std::uint32_t>(y),
                                              static_cast<std::uint32_t>(z)});
    return edits;
}

std::vector<VoxelEdit> make_sphere_edits(std::uint32_t radius) {
    return make_voxel_sphere_edits(centered_sphere(), radius);
}

std::vector<VoxelEdit> make_half_overlap_sphere_edits(std::uint32_t radius) {
    return make_voxel_sphere_edits(half_overlap_sphere(radius), radius);
}

std::uint32_t next_random(std::uint32_t &state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

std::vector<VoxelEdit> make_random_edits(std::uint32_t count) {
    std::uint32_t random_state = 0x9e3779b9u;
    std::vector<VoxelEdit> edits;
    edits.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t x =
            next_random(random_state) % algo::svt::cuda::kWorldVoxelCount;
        const std::uint32_t y =
            next_random(random_state) % algo::svt::cuda::kWorldVoxelCount;
        const std::uint32_t z =
            next_random(random_state) % algo::svt::cuda::kWorldVoxelCount;
        edits.push_back(VoxelEdit{x, y, z});
    }
    return edits;
}

template <class T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    bool allocate(std::size_t count, benchmark::State &state,
                  const char *label) {
        const std::size_t allocated_count = count == 0 ? 1 : count;
        const cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&ptr_),
                                              sizeof(T) * allocated_count);
        if (status != cudaSuccess) {
            state.SkipWithError(label);
            return false;
        }
        return true;
    }

    [[nodiscard]] T *get() const { return ptr_; }

  private:
    T *ptr_ = nullptr;
};

template <class T>
bool copy_to_device(T *dst, const std::vector<T> &src, benchmark::State &state,
                    const char *label) {
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
bool copy_scalar_from_device(T *src, T *dst, benchmark::State &state,
                             const char *label) {
    const cudaError_t status =
        cudaMemcpy(dst, src, sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        state.SkipWithError(label);
        return false;
    }
    return true;
}

bool create_events(cudaEvent_t *start, cudaEvent_t *stop,
                   benchmark::State &state) {
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

bool record_start(cudaEvent_t start, benchmark::State &state) {
    if (cudaEventRecord(start) != cudaSuccess) {
        state.SkipWithError("cudaEventRecord(start) failed");
        return false;
    }
    return true;
}

bool record_elapsed(cudaEvent_t start, cudaEvent_t stop,
                    benchmark::State &state) {
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

void set_voxel_edit_counters(benchmark::State &state, std::uint32_t count) {
    state.counters["voxel_count"] = static_cast<double>(count);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count) *
                            static_cast<int64_t>(sizeof(VoxelEdit)));
}

template <class EditConfig> struct AllocationPathTraits;

template <class Allocation, class Dispatch>
struct AllocationPathTraits<algo::svt::cuda::EditConfig<Allocation, Dispatch>> {
    using AllocationImpl = algo::svt::cuda::detail::allocation_impl<Allocation>;
    using Workspace = typename AllocationImpl::Workspace;
    using CollapseWorkspace = algo::svt::cuda::detail::collapse::Workspace;

    static std::size_t workspace_size(std::uint32_t capacity) {
        return AllocationImpl::workspace_size(capacity);
    }

    static Workspace create_workspace(void *workspace, std::uint32_t capacity) {
        return AllocationImpl::create_workspace(workspace, capacity);
    }

    static cudaError_t allocate(algo::svt::cuda::DeviceGpuSvo svo,
                                const algo::svt::cuda::LeafMask *leaf_masks,
                                const std::uint32_t *leaf_count,
                                std::uint32_t capacity, Workspace &workspace) {
        return AllocationImpl::allocate_paths(svo, leaf_masks, leaf_count,
                                              capacity, workspace, nullptr);
    }

    static cudaError_t
    apply_leaf_masks(algo::svt::cuda::DeviceGpuSvo svo,
                     const algo::svt::cuda::LeafMask *leaf_masks,
                     const std::uint32_t *leaf_count, std::uint32_t capacity,
                     algo::svt::cuda::EditOp op) {
        return algo::svt::cuda::detail::apply_leaf_masks(
            svo, leaf_masks, leaf_count, capacity, op, nullptr);
    }

    static cudaError_t collapse(algo::svt::cuda::DeviceGpuSvo svo,
                                const algo::svt::cuda::LeafMask *leaf_masks,
                                const std::uint32_t *leaf_count,
                                std::uint32_t capacity,
                                CollapseWorkspace &workspace) {
        return algo::svt::cuda::detail::collapse_uniform_paths(
            svo, leaf_masks, leaf_count, capacity, workspace, nullptr);
    }
};

void BM_CudaSvtPhaseBuildLeafMasks(benchmark::State &state,
                                   EditGenerator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto edits =
        make_edits(static_cast<std::uint32_t>(state.range(0)));
    const auto count = static_cast<std::uint32_t>(edits.size());

    DeviceBuffer<VoxelEdit> d_edits;
    DeviceBuffer<algo::svt::cuda::LeafMask> d_leaf_masks;
    DeviceBuffer<std::uint32_t> d_leaf_count;
    DeviceBuffer<std::byte> workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t workspace_size =
        algo::svt::cuda::voxel_edits_to_leaf_masks_workspace_size(count);
    if (!d_edits.allocate(count, state, "cudaMalloc failed for voxel edits") ||
        !d_leaf_masks.allocate(count, state,
                               "cudaMalloc failed for leaf masks") ||
        !d_leaf_count.allocate(1, state, "cudaMalloc failed for leaf count") ||
        !workspace.allocate(workspace_size, state,
                            "cudaMalloc failed for workspace") ||
        !copy_to_device(d_edits.get(), edits, state,
                        "cudaMemcpy failed for voxel edits") ||
        !create_events(&start, &stop, state)) {
        destroy_events(start, stop);
        return;
    }

    for (auto _ : state) {
        if (!record_start(start, state))
            break;
        if (algo::svt::cuda::voxel_edits_to_leaf_masks(
                d_edits.get(), count, d_leaf_masks.get(), d_leaf_count.get(),
                workspace.get(), workspace_size) != cudaSuccess) {
            state.SkipWithError("voxel_edits_to_leaf_masks failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
    }

    set_voxel_edit_counters(state, count);
    destroy_events(start, stop);
}

template <class EditConfig>
void BM_CudaSvtPhaseApplyLeafMasks(benchmark::State &state,
                                   EditGenerator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto edits =
        make_edits(static_cast<std::uint32_t>(state.range(0)));
    const auto count = static_cast<std::uint32_t>(edits.size());

    DeviceBuffer<VoxelEdit> d_edits;
    DeviceBuffer<algo::svt::cuda::LeafMask> d_leaf_masks;
    DeviceBuffer<std::uint32_t> d_leaf_count;
    DeviceBuffer<std::byte> leaf_workspace;
    DeviceBuffer<std::byte> edit_workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t leaf_workspace_size =
        algo::svt::cuda::voxel_edits_to_leaf_masks_workspace_size(count);
    if (!d_edits.allocate(count, state, "cudaMalloc failed for voxel edits") ||
        !d_leaf_masks.allocate(count, state,
                               "cudaMalloc failed for leaf masks") ||
        !d_leaf_count.allocate(1, state, "cudaMalloc failed for leaf count") ||
        !leaf_workspace.allocate(leaf_workspace_size, state,
                                 "cudaMalloc failed for leaf workspace") ||
        !copy_to_device(d_edits.get(), edits, state,
                        "cudaMemcpy failed for voxel edits")) {
        return;
    }

    if (algo::svt::cuda::voxel_edits_to_leaf_masks(
            d_edits.get(), count, d_leaf_masks.get(), d_leaf_count.get(),
            leaf_workspace.get(), leaf_workspace_size) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        state.SkipWithError("voxel_edits_to_leaf_masks setup failed");
        return;
    }

    std::uint32_t host_leaf_count = 0;
    if (!copy_scalar_from_device(d_leaf_count.get(), &host_leaf_count, state,
                                 "cudaMemcpy failed for leaf count")) {
        return;
    }

    const std::size_t edit_workspace_size =
        AllocationPathTraits<EditConfig>::workspace_size(host_leaf_count);
    if (!edit_workspace.allocate(edit_workspace_size, state,
                                 "cudaMalloc failed for edit workspace") ||
        !create_events(&start, &stop, state)) {
        destroy_events(start, stop);
        return;
    }

    typename AllocationPathTraits<EditConfig>::Workspace typed_workspace =
        AllocationPathTraits<EditConfig>::create_workspace(edit_workspace.get(),
                                                           host_leaf_count);

    BenchmarkGpuSvo svo;
    if (svo.status() != cudaSuccess) {
        state.SkipWithError("GpuSvo allocation failed");
        destroy_events(start, stop);
        return;
    }

    for (auto _ : state) {
        if (algo::svt::cuda::reset_svo(svo.view()) != cudaSuccess ||
            AllocationPathTraits<EditConfig>::allocate(
                svo.view(), d_leaf_masks.get(), d_leaf_count.get(),
                host_leaf_count, typed_workspace) != cudaSuccess) {
            state.SkipWithError("apply leaf masks setup failed");
            break;
        }
        if (!record_start(start, state))
            break;
        if (AllocationPathTraits<EditConfig>::apply_leaf_masks(
                svo.view(), d_leaf_masks.get(), d_leaf_count.get(),
                host_leaf_count,
                algo::svt::cuda::EditOp::Place) != cudaSuccess) {
            state.SkipWithError("apply_leaf_masks failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
    }

    state.counters["leaf_count"] = static_cast<double>(host_leaf_count);
    state.counters["capacity"] = static_cast<double>(host_leaf_count);
    set_voxel_edit_counters(state, count);
    destroy_events(start, stop);
}

template <class EditConfig>
void BM_CudaSvtPhaseCollapseUniformPaths(benchmark::State &state,
                                         EditGenerator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto edits =
        make_edits(static_cast<std::uint32_t>(state.range(0)));
    const auto count = static_cast<std::uint32_t>(edits.size());

    DeviceBuffer<VoxelEdit> d_edits;
    DeviceBuffer<algo::svt::cuda::LeafMask> d_leaf_masks;
    DeviceBuffer<std::uint32_t> d_leaf_count;
    DeviceBuffer<std::byte> leaf_workspace;
    DeviceBuffer<std::byte> edit_workspace;
    DeviceBuffer<std::byte> collapse_workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t leaf_workspace_size =
        algo::svt::cuda::voxel_edits_to_leaf_masks_workspace_size(count);
    if (!d_edits.allocate(count, state, "cudaMalloc failed for voxel edits") ||
        !d_leaf_masks.allocate(count, state,
                               "cudaMalloc failed for leaf masks") ||
        !d_leaf_count.allocate(1, state, "cudaMalloc failed for leaf count") ||
        !leaf_workspace.allocate(leaf_workspace_size, state,
                                 "cudaMalloc failed for leaf workspace") ||
        !copy_to_device(d_edits.get(), edits, state,
                        "cudaMemcpy failed for voxel edits")) {
        return;
    }

    if (algo::svt::cuda::voxel_edits_to_leaf_masks(
            d_edits.get(), count, d_leaf_masks.get(), d_leaf_count.get(),
            leaf_workspace.get(), leaf_workspace_size) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        state.SkipWithError("voxel_edits_to_leaf_masks setup failed");
        return;
    }

    std::uint32_t host_leaf_count = 0;
    if (!copy_scalar_from_device(d_leaf_count.get(), &host_leaf_count, state,
                                 "cudaMemcpy failed for leaf count")) {
        return;
    }

    const std::size_t edit_workspace_size =
        AllocationPathTraits<EditConfig>::workspace_size(host_leaf_count);
    const std::size_t collapse_workspace_size =
        algo::svt::cuda::detail::collapse_workspace_size(host_leaf_count);
    if (!edit_workspace.allocate(edit_workspace_size, state,
                                 "cudaMalloc failed for edit workspace") ||
        !collapse_workspace.allocate(
            collapse_workspace_size, state,
            "cudaMalloc failed for collapse workspace") ||
        !create_events(&start, &stop, state)) {
        destroy_events(start, stop);
        return;
    }

    typename AllocationPathTraits<EditConfig>::Workspace typed_workspace =
        AllocationPathTraits<EditConfig>::create_workspace(edit_workspace.get(),
                                                           host_leaf_count);
    typename AllocationPathTraits<EditConfig>::CollapseWorkspace
        typed_collapse_workspace =
            algo::svt::cuda::detail::create_collapse_workspace(
                collapse_workspace.get(), host_leaf_count);

    BenchmarkGpuSvo svo;
    if (svo.status() != cudaSuccess) {
        state.SkipWithError("GpuSvo allocation failed");
        destroy_events(start, stop);
        return;
    }

    for (auto _ : state) {
        if (algo::svt::cuda::reset_svo(svo.view()) != cudaSuccess ||
            AllocationPathTraits<EditConfig>::allocate(
                svo.view(), d_leaf_masks.get(), d_leaf_count.get(),
                host_leaf_count, typed_workspace) != cudaSuccess ||
            AllocationPathTraits<EditConfig>::apply_leaf_masks(
                svo.view(), d_leaf_masks.get(), d_leaf_count.get(),
                host_leaf_count,
                algo::svt::cuda::EditOp::Place) != cudaSuccess) {
            state.SkipWithError("collapse setup failed");
            break;
        }
        if (!record_start(start, state))
            break;
        if (AllocationPathTraits<EditConfig>::collapse(
                svo.view(), d_leaf_masks.get(), d_leaf_count.get(),
                host_leaf_count, typed_collapse_workspace) != cudaSuccess) {
            state.SkipWithError("collapse_uniform_paths failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
    }

    state.counters["leaf_count"] = static_cast<double>(host_leaf_count);
    state.counters["capacity"] = static_cast<double>(host_leaf_count);
    set_voxel_edit_counters(state, count);
    destroy_events(start, stop);
}

template <class EditConfig>
void BM_CudaSvtAllocatePaths(benchmark::State &state,
                             EditGenerator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto edits =
        make_edits(static_cast<std::uint32_t>(state.range(0)));
    const auto count = static_cast<std::uint32_t>(edits.size());

    DeviceBuffer<VoxelEdit> d_edits;
    DeviceBuffer<algo::svt::cuda::LeafMask> d_leaf_masks;
    DeviceBuffer<std::uint32_t> d_leaf_count;
    DeviceBuffer<std::byte> leaf_workspace;
    DeviceBuffer<std::byte> allocation_workspace;

    const std::size_t leaf_workspace_size =
        algo::svt::cuda::voxel_edits_to_leaf_masks_workspace_size(count);
    if (!d_edits.allocate(count, state, "cudaMalloc failed for voxel edits") ||
        !d_leaf_masks.allocate(count, state,
                               "cudaMalloc failed for leaf masks") ||
        !d_leaf_count.allocate(1, state, "cudaMalloc failed for leaf count") ||
        !leaf_workspace.allocate(leaf_workspace_size, state,
                                 "cudaMalloc failed for leaf workspace") ||
        !copy_to_device(d_edits.get(), edits, state,
                        "cudaMemcpy failed for voxel edits")) {
        return;
    }

    if (algo::svt::cuda::voxel_edits_to_leaf_masks(
            d_edits.get(), count, d_leaf_masks.get(), d_leaf_count.get(),
            leaf_workspace.get(), leaf_workspace_size) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        state.SkipWithError("voxel_edits_to_leaf_masks setup failed");
        return;
    }

    std::uint32_t host_leaf_count = 0;
    std::uint32_t setup_capacity = 0;
    if (algo::svt::cuda::detail::resolve_dispatch_capacity<
            algo::svt::cuda::HostLeafCountDispatch>(d_leaf_count.get(), count,
                                                    &host_leaf_count,
                                                    nullptr) != cudaSuccess) {
        state.SkipWithError("resolve leaf count setup failed");
        return;
    }
    if (algo::svt::cuda::detail::resolve_dispatch_capacity<
            typename EditConfig::dispatch>(d_leaf_count.get(), count,
                                           &setup_capacity,
                                           nullptr) != cudaSuccess) {
        state.SkipWithError("resolve dispatch capacity setup failed");
        return;
    }

    const std::size_t allocation_workspace_size =
        AllocationPathTraits<EditConfig>::workspace_size(count);
    if (!allocation_workspace.allocate(
            allocation_workspace_size, state,
            "cudaMalloc failed for allocation workspace")) {
        return;
    }

    BenchmarkGpuSvo svo;
    if (svo.status() != cudaSuccess) {
        state.SkipWithError("GpuSvo allocation failed");
        return;
    }

    for (auto _ : state) {
        if (algo::svt::cuda::reset_svo(svo.view()) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            state.SkipWithError("reset_svo failed");
            break;
        }

        const auto start = std::chrono::steady_clock::now();

        std::uint32_t capacity = 0;
        if (algo::svt::cuda::detail::resolve_dispatch_capacity<
                typename EditConfig::dispatch>(
                d_leaf_count.get(), count, &capacity, nullptr) != cudaSuccess) {
            state.SkipWithError("resolve dispatch capacity failed");
            break;
        }

        typename AllocationPathTraits<EditConfig>::Workspace typed_workspace =
            AllocationPathTraits<EditConfig>::create_workspace(
                allocation_workspace.get(), capacity);

        if (AllocationPathTraits<EditConfig>::allocate(
                svo.view(), d_leaf_masks.get(), d_leaf_count.get(), capacity,
                typed_workspace) != cudaSuccess) {
            state.SkipWithError("allocate_paths failed");
            break;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            state.SkipWithError("allocate_paths synchronize failed");
            break;
        }

        const auto stop = std::chrono::steady_clock::now();
        state.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        benchmark::DoNotOptimize(capacity);
    }

    state.counters["leaf_count"] = static_cast<double>(host_leaf_count);
    state.counters["capacity"] = static_cast<double>(setup_capacity);
    set_voxel_edit_counters(state, count);
}

void apply_svt_args(benchmark::internal::Benchmark *bench) {
    bench->Arg(1 << 22)->Arg(1 << 24);
}

void apply_svt_sphere_args(benchmark::internal::Benchmark *bench) {
    bench->Arg(128);
}

#define SVT_PHASE_ALLOCATE_CONFIGS(X)                                                  \
    X(ScanDepthwiseVoxelCountDispatch,                                                 \
      algo::svt::cuda::EditConfig<algo::svt::cuda::ScanDepthwiseAllocation,            \
                                  algo::svt::cuda::VoxelCountDispatch>)                \
    X(ScanDepthwiseHostLeafCountDispatch,                                              \
      algo::svt::cuda::EditConfig<algo::svt::cuda::ScanDepthwiseAllocation,            \
                                  algo::svt::cuda::HostLeafCountDispatch>)             \
    X(CachedDepthwiseVoxelCountDispatch,                                               \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CachedDepthwiseAllocation,          \
                                  algo::svt::cuda::VoxelCountDispatch>)                \
    X(CachedDepthwiseHostLeafCountDispatch,                                            \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CachedDepthwiseAllocation,          \
                                  algo::svt::cuda::HostLeafCountDispatch>)             \
    X(PlainDepthwiseHostLeafCountDispatch,                                             \
      algo::svt::cuda::EditConfig<algo::svt::cuda::PlainDepthwiseAllocation,           \
                                  algo::svt::cuda::HostLeafCountDispatch>)             \
    X(AllDepthVoxelCountDispatch,                                                      \
      algo::svt::cuda::EditConfig<algo::svt::cuda::AllDepthAllocation,                 \
                                  algo::svt::cuda::VoxelCountDispatch>)                \
    X(AllDepthHostLeafCountDispatch,                                                   \
      algo::svt::cuda::EditConfig<algo::svt::cuda::AllDepthAllocation,                 \
                                  algo::svt::cuda::HostLeafCountDispatch>)             \
    X(CompactAllDepthStartDepthVoxelCountDispatch,                                     \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Threadwise, algo::svt::cuda::StoreStartDepth>,          \
          algo::svt::cuda::VoxelCountDispatch>)                                        \
    X(CompactAllDepthStartDepthHostLeafCountDispatch,                                  \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Threadwise, algo::svt::cuda::StoreStartDepth>,          \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthStartDepthNodewiseOffsetSearchHostLeafCountDispatch,              \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Threadwise, algo::svt::cuda::StoreStartDepth,           \
              algo::svt::cuda::Nodewise<algo::svt::cuda::OffsetSearch>>,               \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthStartDepthNodewiseExplicitRequestsHostLeafCountDispatch,          \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Threadwise, algo::svt::cuda::StoreStartDepth,           \
              algo::svt::cuda::Nodewise<algo::svt::cuda::ExplicitRequests>>,           \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthVoxelCountDispatch,                                               \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<          \
                                      algo::svt::cuda::Threadwise>,                    \
                                  algo::svt::cuda::VoxelCountDispatch>)                \
    X(CompactAllDepthHostLeafCountDispatch,                                            \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<          \
                                      algo::svt::cuda::Threadwise>,                    \
                                  algo::svt::cuda::HostLeafCountDispatch>)             \
    X(CompactAllDepthNodewiseOffsetSearchHostLeafCountDispatch,                        \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Threadwise, algo::svt::cuda::RecoverStartDepth,         \
              algo::svt::cuda::Nodewise<algo::svt::cuda::OffsetSearch>>,               \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthNodewiseExplicitRequestsHostLeafCountDispatch,                    \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Threadwise, algo::svt::cuda::RecoverStartDepth,         \
              algo::svt::cuda::Nodewise<algo::svt::cuda::ExplicitRequests>>,           \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthStartDepthChildwiseVoxelCountDispatch,                            \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Childwise, algo::svt::cuda::StoreStartDepth>,           \
          algo::svt::cuda::VoxelCountDispatch>)                                        \
    X(CompactAllDepthStartDepthChildwiseHostLeafCountDispatch,                         \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Childwise, algo::svt::cuda::StoreStartDepth>,           \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthStartDepthChildwiseNodewiseOffsetSearchHostLeafCountDispatch,     \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Childwise, algo::svt::cuda::StoreStartDepth,            \
              algo::svt::cuda::Nodewise<algo::svt::cuda::OffsetSearch>>,               \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthStartDepthChildwiseNodewiseExplicitRequestsHostLeafCountDispatch, \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Childwise, algo::svt::cuda::StoreStartDepth,            \
              algo::svt::cuda::Nodewise<algo::svt::cuda::ExplicitRequests>>,           \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthChildwiseVoxelCountDispatch,                                      \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<          \
                                      algo::svt::cuda::Childwise>,                     \
                                  algo::svt::cuda::VoxelCountDispatch>)                \
    X(CompactAllDepthChildwiseHostLeafCountDispatch,                                   \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<          \
                                      algo::svt::cuda::Childwise>,                     \
                                  algo::svt::cuda::HostLeafCountDispatch>)             \
    X(CompactAllDepthChildwiseNodewiseOffsetSearchHostLeafCountDispatch,               \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Childwise, algo::svt::cuda::RecoverStartDepth,          \
              algo::svt::cuda::Nodewise<algo::svt::cuda::OffsetSearch>>,               \
          algo::svt::cuda::HostLeafCountDispatch>)                                     \
    X(CompactAllDepthChildwiseNodewiseExplicitRequestsHostLeafCountDispatch,           \
      algo::svt::cuda::EditConfig<                                                     \
          algo::svt::cuda::CompactAllDepthAllocation<                                  \
              algo::svt::cuda::Childwise, algo::svt::cuda::RecoverStartDepth,          \
              algo::svt::cuda::Nodewise<algo::svt::cuda::ExplicitRequests>>,           \
          algo::svt::cuda::HostLeafCountDispatch>)

#define DEFINE_ALLOCATE_PATHS_BENCHMARK(Name, ...)                             \
    void BM_CudaSvtAllocatePathsSphere##Name(benchmark::State &state) {        \
        BM_CudaSvtAllocatePaths<__VA_ARGS__>(state, make_sphere_edits);        \
    }                                                                          \
                                                                               \
    void BM_CudaSvtAllocatePathsSphereHalfOverlap##Name(benchmark::State       \
                                                            &state) {          \
        BM_CudaSvtAllocatePaths<__VA_ARGS__>(                                  \
            state, make_half_overlap_sphere_edits);                            \
    }                                                                          \
                                                                               \
    void BM_CudaSvtAllocatePathsRandom##Name(benchmark::State &state) {        \
        BM_CudaSvtAllocatePaths<__VA_ARGS__>(state, make_random_edits);        \
    }

SVT_PHASE_ALLOCATE_CONFIGS(DEFINE_ALLOCATE_PATHS_BENCHMARK)

#undef DEFINE_ALLOCATE_PATHS_BENCHMARK

void BM_CudaSvtPhaseBuildLeafMasksSphere(benchmark::State &state) {
    BM_CudaSvtPhaseBuildLeafMasks(state, make_sphere_edits);
}

void BM_CudaSvtPhaseBuildLeafMasksSphereHalfOverlap(benchmark::State &state) {
    BM_CudaSvtPhaseBuildLeafMasks(state, make_half_overlap_sphere_edits);
}

void BM_CudaSvtPhaseBuildLeafMasksRandom(benchmark::State &state) {
    BM_CudaSvtPhaseBuildLeafMasks(state, make_random_edits);
}

using RepresentativePhaseConfig = algo::svt::cuda::EditConfig<
    algo::svt::cuda::CompactAllDepthAllocation<algo::svt::cuda::Threadwise>,
    algo::svt::cuda::HostLeafCountDispatch>;

void BM_CudaSvtPhaseApplyLeafMasksSphere(benchmark::State &state) {
    BM_CudaSvtPhaseApplyLeafMasks<RepresentativePhaseConfig>(state,
                                                             make_sphere_edits);
}

void BM_CudaSvtPhaseApplyLeafMasksSphereHalfOverlap(benchmark::State &state) {
    BM_CudaSvtPhaseApplyLeafMasks<RepresentativePhaseConfig>(
        state, make_half_overlap_sphere_edits);
}

void BM_CudaSvtPhaseApplyLeafMasksRandom(benchmark::State &state) {
    BM_CudaSvtPhaseApplyLeafMasks<RepresentativePhaseConfig>(state,
                                                             make_random_edits);
}

void BM_CudaSvtPhaseCollapseUniformPathsSphere(benchmark::State &state) {
    BM_CudaSvtPhaseCollapseUniformPaths<RepresentativePhaseConfig>(
        state, make_sphere_edits);
}

void BM_CudaSvtPhaseCollapseUniformPathsSphereHalfOverlap(
    benchmark::State &state) {
    BM_CudaSvtPhaseCollapseUniformPaths<RepresentativePhaseConfig>(
        state, make_half_overlap_sphere_edits);
}

void BM_CudaSvtPhaseCollapseUniformPathsRandom(benchmark::State &state) {
    BM_CudaSvtPhaseCollapseUniformPaths<RepresentativePhaseConfig>(
        state, make_random_edits);
}

BENCHMARK(BM_CudaSvtPhaseBuildLeafMasksSphere)
    ->UseManualTime()
    ->Apply(apply_svt_sphere_args);
BENCHMARK(BM_CudaSvtPhaseBuildLeafMasksSphereHalfOverlap)
    ->UseManualTime()
    ->Apply(apply_svt_sphere_args);
BENCHMARK(BM_CudaSvtPhaseBuildLeafMasksRandom)
    ->UseManualTime()
    ->Apply(apply_svt_args);
BENCHMARK(BM_CudaSvtPhaseApplyLeafMasksSphere)
    ->UseManualTime()
    ->Apply(apply_svt_sphere_args);
BENCHMARK(BM_CudaSvtPhaseApplyLeafMasksSphereHalfOverlap)
    ->UseManualTime()
    ->Apply(apply_svt_sphere_args);
BENCHMARK(BM_CudaSvtPhaseApplyLeafMasksRandom)
    ->UseManualTime()
    ->Apply(apply_svt_args);
BENCHMARK(BM_CudaSvtPhaseCollapseUniformPathsSphere)
    ->UseManualTime()
    ->Apply(apply_svt_sphere_args);
BENCHMARK(BM_CudaSvtPhaseCollapseUniformPathsSphereHalfOverlap)
    ->UseManualTime()
    ->Apply(apply_svt_sphere_args);
BENCHMARK(BM_CudaSvtPhaseCollapseUniformPathsRandom)
    ->UseManualTime()
    ->Apply(apply_svt_args);

#define REGISTER_ALLOCATE_PATHS_BENCHMARK(Name, ...)                           \
    BENCHMARK(BM_CudaSvtAllocatePathsSphere##Name)                             \
        ->UseManualTime()                                                      \
        ->Apply(apply_svt_sphere_args);                                        \
    BENCHMARK(BM_CudaSvtAllocatePathsSphereHalfOverlap##Name)                  \
        ->UseManualTime()                                                      \
        ->Apply(apply_svt_sphere_args);                                        \
    BENCHMARK(BM_CudaSvtAllocatePathsRandom##Name)                             \
        ->UseManualTime()                                                      \
        ->Apply(apply_svt_args);

SVT_PHASE_ALLOCATE_CONFIGS(REGISTER_ALLOCATE_PATHS_BENCHMARK)

#undef REGISTER_ALLOCATE_PATHS_BENCHMARK
#undef SVT_PHASE_ALLOCATE_CONFIGS

} // namespace
