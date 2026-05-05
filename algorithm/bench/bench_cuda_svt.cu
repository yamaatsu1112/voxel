#include <algo/svt/cuda.cuh>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using algo::svt::cuda::VoxelEdit;
using EditGenerator = std::vector<VoxelEdit> (*)(std::uint32_t);
using BenchmarkGpuSvo = algo::svt::cuda::GpuSvo<(1u << 26), (1u << 24)>;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::uint32_t block_count(std::uint32_t count, std::uint32_t block_size) {
    return (count + block_size - 1u) / block_size;
}

std::uint32_t dense_side_for(std::uint32_t count) {
    std::uint32_t side = static_cast<std::uint32_t>(
        std::ceil(std::cbrt(static_cast<double>(count))));
    side = std::max(side, 1u);
    while (side * side * side < count)
        ++side;
    return side;
}

std::vector<VoxelEdit> make_dense_edits(std::uint32_t count) {
    const std::uint32_t side = dense_side_for(count);
    std::vector<VoxelEdit> edits;
    edits.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t x = i % side;
        const std::uint32_t y = (i / side) % side;
        const std::uint32_t z = i / (side * side);
        edits.push_back(VoxelEdit{x, y, z});
    }
    return edits;
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
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count) *
                            static_cast<int64_t>(sizeof(VoxelEdit)));
}

__global__ void query_voxels_kernel(algo::svt::cuda::DeviceGpuSvo svo,
                                    const VoxelEdit *queries,
                                    std::uint8_t *results,
                                    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const VoxelEdit query = queries[index];
    results[index] =
        algo::svt::cuda::get_voxel(svo, query.x, query.y, query.z) ? 1u : 0u;
}

void BM_CudaSvtVoxelEditsToLeafMasks(benchmark::State &state,
                                     EditGenerator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    const auto edits = make_edits(count);

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
void BM_CudaSvtPlaceVoxelEdits(benchmark::State &state,
                               EditGenerator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    const auto edits = make_edits(count);

    DeviceBuffer<VoxelEdit> d_edits;
    DeviceBuffer<std::byte> workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<EditConfig>(count);
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
        if (algo::svt::cuda::place_voxel_edits<EditConfig>(
                svo.view(), d_edits.get(), count, workspace.get(),
                workspace_size) != cudaSuccess) {
            state.SkipWithError("place_voxel_edits failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
    }

    set_voxel_edit_counters(state, count);
    destroy_events(start, stop);
}

template <class EditConfig>
void BM_CudaSvtDestroyVoxelEdits(benchmark::State &state,
                                 EditGenerator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    const auto edits = make_edits(count);

    DeviceBuffer<VoxelEdit> d_edits;
    DeviceBuffer<std::byte> place_workspace;
    DeviceBuffer<std::byte> destroy_workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t place_workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<EditConfig>(count);
    const std::size_t destroy_workspace_size =
        algo::svt::cuda::apply_voxel_edits_workspace_size<EditConfig>(count);
    BenchmarkGpuSvo svo;
    if (svo.status() != cudaSuccess) {
        state.SkipWithError("GpuSvo allocation failed");
        return;
    }

    if (!d_edits.allocate(count, state, "cudaMalloc failed for voxel edits") ||
        !place_workspace.allocate(place_workspace_size, state,
                                  "cudaMalloc failed for place workspace") ||
        !destroy_workspace.allocate(
            destroy_workspace_size, state,
            "cudaMalloc failed for destroy workspace") ||
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
        if (algo::svt::cuda::place_voxel_edits<EditConfig>(
                svo.view(), d_edits.get(), count, place_workspace.get(),
                place_workspace_size) != cudaSuccess) {
            state.SkipWithError("place_voxel_edits setup failed");
            break;
        }
        if (!record_start(start, state))
            break;
        if (algo::svt::cuda::destroy_voxel_edits<EditConfig>(
                svo.view(), d_edits.get(), count, destroy_workspace.get(),
                destroy_workspace_size) != cudaSuccess) {
            state.SkipWithError("destroy_voxel_edits failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
    }

    set_voxel_edit_counters(state, count);
    destroy_events(start, stop);
}

void BM_CudaSvtGetVoxel(benchmark::State &state, EditGenerator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    const auto edits = make_edits(count);

    DeviceBuffer<VoxelEdit> d_queries;
    DeviceBuffer<std::uint8_t> d_results;
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

    if (!d_queries.allocate(count, state, "cudaMalloc failed for queries") ||
        !d_results.allocate(count, state, "cudaMalloc failed for results") ||
        !workspace.allocate(workspace_size, state,
                            "cudaMalloc failed for workspace") ||
        !copy_to_device(d_queries.get(), edits, state,
                        "cudaMemcpy failed for queries") ||
        !create_events(&start, &stop, state)) {
        destroy_events(start, stop);
        return;
    }

    if (algo::svt::cuda::reset_svo(svo.view()) != cudaSuccess ||
        algo::svt::cuda::place_voxel_edits(svo.view(), d_queries.get(), count,
                                           workspace.get(),
                                           workspace_size) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        state.SkipWithError("SVO setup failed");
        destroy_events(start, stop);
        return;
    }

    constexpr std::uint32_t kBlockSize = 256;
    for (auto _ : state) {
        if (!record_start(start, state))
            break;
        query_voxels_kernel<<<block_count(count, kBlockSize), kBlockSize>>>(
            svo.view(), d_queries.get(), d_results.get(), count);
        if (cudaGetLastError() != cudaSuccess) {
            state.SkipWithError("query_voxels_kernel launch failed");
            break;
        }
        if (!record_elapsed(start, stop, state))
            break;
    }

    benchmark::DoNotOptimize(d_results.get());
    set_voxel_edit_counters(state, count);
    destroy_events(start, stop);
}

void apply_svt_args(benchmark::internal::Benchmark *bench) {
    bench->Arg(1 << 16)->Arg(1 << 18)->Arg(1 << 20)->Arg(1 << 22)->Arg(1 << 24);
}

#define SVT_EDIT_CONFIGS(X)                                                    \
    X(ScanDepthwise,                                                           \
      algo::svt::cuda::EditConfig<algo::svt::cuda::ScanDepthwiseAllocation,    \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(ScanDepthwiseHostLeafCountDispatch,                                      \
      algo::svt::cuda::EditConfig<algo::svt::cuda::ScanDepthwiseAllocation,    \
                                  algo::svt::cuda::HostLeafCountDispatch>)     \
    X(CachedDepthwise,                                                         \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CachedDepthwiseAllocation,  \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(CachedDepthwiseHostLeafCountDispatch,                                    \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CachedDepthwiseAllocation,  \
                                  algo::svt::cuda::HostLeafCountDispatch>)     \
    X(AllDepth,                                                                \
      algo::svt::cuda::EditConfig<algo::svt::cuda::AllDepthAllocation,         \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(AllDepthHostLeafCountDispatch,                                           \
      algo::svt::cuda::EditConfig<algo::svt::cuda::AllDepthAllocation,         \
                                  algo::svt::cuda::HostLeafCountDispatch>)     \
    X(CompactAllDepthStartDepth,                                               \
      algo::svt::cuda::EditConfig<                                             \
          algo::svt::cuda::CompactAllDepthAllocation<                          \
              algo::svt::cuda::Threadwise, algo::svt::cuda::StoreStartDepth>,  \
          algo::svt::cuda::VoxelCountDispatch>)                                \
    X(CompactAllDepthStartDepthHostLeafCountDispatch,                          \
      algo::svt::cuda::EditConfig<                                             \
          algo::svt::cuda::CompactAllDepthAllocation<                          \
              algo::svt::cuda::Threadwise, algo::svt::cuda::StoreStartDepth>,  \
          algo::svt::cuda::HostLeafCountDispatch>)                             \
    X(CompactAllDepth,                                                         \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<  \
                                      algo::svt::cuda::Threadwise>,            \
                                  algo::svt::cuda::VoxelCountDispatch>)        \
    X(CompactAllDepthHostLeafCountDispatch,                                    \
      algo::svt::cuda::EditConfig<algo::svt::cuda::CompactAllDepthAllocation<  \
                                      algo::svt::cuda::Threadwise>,            \
                                  algo::svt::cuda::HostLeafCountDispatch>)

#define DEFINE_PLACE_BENCHMARK(Name, ...)                                      \
    void BM_CudaSvtPlaceVoxelEditsDense##Name(benchmark::State &state) {       \
        BM_CudaSvtPlaceVoxelEdits<__VA_ARGS__>(state, make_dense_edits);       \
    }                                                                          \
                                                                               \
    void BM_CudaSvtPlaceVoxelEditsRandom##Name(benchmark::State &state) {      \
        BM_CudaSvtPlaceVoxelEdits<__VA_ARGS__>(state, make_random_edits);      \
    }

SVT_EDIT_CONFIGS(DEFINE_PLACE_BENCHMARK)

#undef DEFINE_PLACE_BENCHMARK

#define DEFINE_DESTROY_BENCHMARK(Name, ...)                                    \
    void BM_CudaSvtDestroyVoxelEditsDense##Name(benchmark::State &state) {     \
        BM_CudaSvtDestroyVoxelEdits<__VA_ARGS__>(state, make_dense_edits);     \
    }                                                                          \
                                                                               \
    void BM_CudaSvtDestroyVoxelEditsRandom##Name(benchmark::State &state) {    \
        BM_CudaSvtDestroyVoxelEdits<__VA_ARGS__>(state, make_random_edits);    \
    }

SVT_EDIT_CONFIGS(DEFINE_DESTROY_BENCHMARK)

#undef DEFINE_DESTROY_BENCHMARK

void BM_CudaSvtVoxelEditsToLeafMasksDense(benchmark::State &state) {
    BM_CudaSvtVoxelEditsToLeafMasks(state, make_dense_edits);
}

void BM_CudaSvtVoxelEditsToLeafMasksRandom(benchmark::State &state) {
    BM_CudaSvtVoxelEditsToLeafMasks(state, make_random_edits);
}

void BM_CudaSvtGetVoxelDense(benchmark::State &state) {
    BM_CudaSvtGetVoxel(state, make_dense_edits);
}

void BM_CudaSvtGetVoxelRandom(benchmark::State &state) {
    BM_CudaSvtGetVoxel(state, make_random_edits);
}

BENCHMARK(BM_CudaSvtVoxelEditsToLeafMasksDense)
    ->UseManualTime()
    ->Apply(apply_svt_args);
BENCHMARK(BM_CudaSvtVoxelEditsToLeafMasksRandom)
    ->UseManualTime()
    ->Apply(apply_svt_args);

#define REGISTER_PLACE_BENCHMARK(Name, ...)                                    \
    BENCHMARK(BM_CudaSvtPlaceVoxelEditsDense##Name)                            \
        ->UseManualTime()                                                      \
        ->Apply(apply_svt_args);                                               \
    BENCHMARK(BM_CudaSvtPlaceVoxelEditsRandom##Name)                           \
        ->UseManualTime()                                                      \
        ->Apply(apply_svt_args);

SVT_EDIT_CONFIGS(REGISTER_PLACE_BENCHMARK)

#undef REGISTER_PLACE_BENCHMARK

#define REGISTER_DESTROY_BENCHMARK(Name, ...)                                  \
    BENCHMARK(BM_CudaSvtDestroyVoxelEditsDense##Name)                          \
        ->UseManualTime()                                                      \
        ->Apply(apply_svt_args);                                               \
    BENCHMARK(BM_CudaSvtDestroyVoxelEditsRandom##Name)                         \
        ->UseManualTime()                                                      \
        ->Apply(apply_svt_args);

SVT_EDIT_CONFIGS(REGISTER_DESTROY_BENCHMARK)

#undef REGISTER_DESTROY_BENCHMARK

BENCHMARK(BM_CudaSvtGetVoxelDense)->UseManualTime()->Apply(apply_svt_args);
BENCHMARK(BM_CudaSvtGetVoxelRandom)->UseManualTime()->Apply(apply_svt_args);

#undef SVT_EDIT_CONFIGS

} // namespace
