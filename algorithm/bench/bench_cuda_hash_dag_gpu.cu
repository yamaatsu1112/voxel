#include <algo/svt/hash_dag_gpu.cuh>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

namespace {

namespace hash_dag_gpu = algo::svt::hash_dag_gpu;

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

std::uint32_t next_random(std::uint32_t &state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
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

struct HashDagGpuBackend {
    using VoxelEdit = hash_dag_gpu::VoxelEdit;
    using Dag = hash_dag_gpu::HashDagGpu<(1u << 16), (1u << 16)>;
    using DeviceDag = hash_dag_gpu::DeviceHashDagGpu;

    static constexpr std::uint32_t kWorldVoxelCount =
        hash_dag_gpu::kWorldVoxelCount;

    static std::size_t workspace_size(std::uint32_t count) {
        return hash_dag_gpu::apply_voxel_edits_workspace_size(count);
    }

    static cudaError_t place(DeviceDag dag, const VoxelEdit *edits,
                             std::uint32_t count, void *workspace,
                             std::size_t workspace_size) {
        return hash_dag_gpu::place_voxel_edits(dag, edits, count, workspace,
                                               workspace_size);
    }

    static cudaError_t destroy(DeviceDag dag, const VoxelEdit *edits,
                               std::uint32_t count, void *workspace,
                               std::size_t workspace_size) {
        return hash_dag_gpu::destroy_voxel_edits(dag, edits, count, workspace,
                                                 workspace_size);
    }
};

template <class Backend>
std::vector<typename Backend::VoxelEdit> make_dense_edits(std::uint32_t count) {
    const std::uint32_t side = dense_side_for(count);
    std::vector<typename Backend::VoxelEdit> edits;
    edits.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t x = i % side;
        const std::uint32_t y = (i / side) % side;
        const std::uint32_t z = i / (side * side);
        edits.push_back(typename Backend::VoxelEdit{x, y, z});
    }
    return edits;
}

template <class Backend>
std::vector<typename Backend::VoxelEdit>
make_random_edits(std::uint32_t count) {
    std::uint32_t random_state = 0x9e3779b9u;
    std::vector<typename Backend::VoxelEdit> edits;
    edits.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t x =
            next_random(random_state) % Backend::kWorldVoxelCount;
        const std::uint32_t y =
            next_random(random_state) % Backend::kWorldVoxelCount;
        const std::uint32_t z =
            next_random(random_state) % Backend::kWorldVoxelCount;
        edits.push_back(typename Backend::VoxelEdit{x, y, z});
    }
    return edits;
}

__global__ void
query_hash_dag_gpu_voxels_kernel(algo::svt::hash_dag_gpu::DeviceHashDagGpu dag,
                                 const HashDagGpuBackend::VoxelEdit *queries,
                                 std::uint8_t *results, std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const HashDagGpuBackend::VoxelEdit query = queries[index];
    results[index] =
        algo::svt::hash_dag_gpu::get_voxel(dag, query.x, query.y, query.z) ? 1u
                                                                           : 0u;
}

template <class Backend>
void set_voxel_edit_counters(benchmark::State &state, std::uint32_t count) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(count) *
        static_cast<int64_t>(sizeof(typename Backend::VoxelEdit)));
}

template <class Backend, class Generator>
void BM_CudaHashDagPlaceVoxelEdits(benchmark::State &state,
                                   Generator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    const auto edits = make_edits(count);

    DeviceBuffer<typename Backend::VoxelEdit> d_edits;
    DeviceBuffer<std::byte> workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t workspace_size = Backend::workspace_size(count);

    try {
        typename Backend::Dag dag;
        if (dag.status() != cudaSuccess) {
            state.SkipWithError("HashDag allocation failed");
            return;
        }

        if (!d_edits.allocate(count, state,
                              "cudaMalloc failed for voxel edits") ||
            !workspace.allocate(workspace_size, state,
                                "cudaMalloc failed for workspace") ||
            !copy_to_device(d_edits.get(), edits, state,
                            "cudaMemcpy failed for voxel edits") ||
            !create_events(&start, &stop, state)) {
            destroy_events(start, stop);
            return;
        }

        for (auto _ : state) {
            if (dag.reset() != cudaSuccess) {
                state.SkipWithError("HashDag reset failed");
                break;
            }
            if (!record_start(start, state))
                break;
            if (Backend::place(dag.view(), d_edits.get(), count,
                               workspace.get(),
                               workspace_size) != cudaSuccess) {
                state.SkipWithError("place_voxel_edits failed");
                break;
            }
            if (!record_elapsed(start, stop, state))
                break;
        }
    } catch (const std::exception &error) {
        state.SkipWithError(error.what());
    }

    set_voxel_edit_counters<Backend>(state, count);
    destroy_events(start, stop);
}

template <class Backend, class Generator>
void BM_CudaHashDagDestroyVoxelEdits(benchmark::State &state,
                                     Generator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    const auto edits = make_edits(count);

    DeviceBuffer<typename Backend::VoxelEdit> d_edits;
    DeviceBuffer<std::byte> place_workspace;
    DeviceBuffer<std::byte> destroy_workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t place_workspace_size = Backend::workspace_size(count);
    const std::size_t destroy_workspace_size = Backend::workspace_size(count);

    try {
        typename Backend::Dag dag;
        if (dag.status() != cudaSuccess) {
            state.SkipWithError("HashDag allocation failed");
            return;
        }

        if (!d_edits.allocate(count, state,
                              "cudaMalloc failed for voxel edits") ||
            !place_workspace.allocate(
                place_workspace_size, state,
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
            if (dag.reset() != cudaSuccess) {
                state.SkipWithError("HashDag reset failed");
                break;
            }
            if (Backend::place(dag.view(), d_edits.get(), count,
                               place_workspace.get(),
                               place_workspace_size) != cudaSuccess) {
                state.SkipWithError("place_voxel_edits setup failed");
                break;
            }
            if (!record_start(start, state))
                break;
            if (Backend::destroy(dag.view(), d_edits.get(), count,
                                 destroy_workspace.get(),
                                 destroy_workspace_size) != cudaSuccess) {
                state.SkipWithError("destroy_voxel_edits failed");
                break;
            }
            if (!record_elapsed(start, stop, state))
                break;
        }
    } catch (const std::exception &error) {
        state.SkipWithError(error.what());
    }

    set_voxel_edit_counters<Backend>(state, count);
    destroy_events(start, stop);
}

template <class Backend, class Generator>
void BM_CudaHashDagGetVoxel(benchmark::State &state, Generator make_edits) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    const auto edits = make_edits(count);

    DeviceBuffer<typename Backend::VoxelEdit> d_queries;
    DeviceBuffer<std::uint8_t> d_results;
    DeviceBuffer<std::byte> workspace;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const std::size_t workspace_size = Backend::workspace_size(count);

    try {
        typename Backend::Dag dag;
        if (dag.status() != cudaSuccess) {
            state.SkipWithError("HashDag allocation failed");
            return;
        }

        if (!d_queries.allocate(count, state,
                                "cudaMalloc failed for queries") ||
            !d_results.allocate(count, state,
                                "cudaMalloc failed for results") ||
            !workspace.allocate(workspace_size, state,
                                "cudaMalloc failed for workspace") ||
            !copy_to_device(d_queries.get(), edits, state,
                            "cudaMemcpy failed for queries") ||
            !create_events(&start, &stop, state)) {
            destroy_events(start, stop);
            return;
        }

        if (dag.reset() != cudaSuccess ||
            Backend::place(dag.view(), d_queries.get(), count, workspace.get(),
                           workspace_size) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            state.SkipWithError("HashDag setup failed");
            destroy_events(start, stop);
            return;
        }

        constexpr std::uint32_t kBlockSize = 256;
        for (auto _ : state) {
            if (!record_start(start, state))
                break;
            query_hash_dag_gpu_voxels_kernel<<<block_count(count, kBlockSize),
                                               kBlockSize>>>(
                dag.view(), d_queries.get(), d_results.get(), count);
            if (cudaGetLastError() != cudaSuccess) {
                state.SkipWithError("query voxels kernel launch failed");
                break;
            }
            if (!record_elapsed(start, stop, state))
                break;
        }
    } catch (const std::exception &error) {
        state.SkipWithError(error.what());
    }

    benchmark::DoNotOptimize(d_results.get());
    set_voxel_edit_counters<Backend>(state, count);
    destroy_events(start, stop);
}

void apply_hash_dag_args(benchmark::internal::Benchmark *bench) {
    bench->Arg(1 << 16)->Arg(1 << 18)->Arg(1 << 20)->Arg(1 << 22)->Arg(1 << 24);
}

void BM_CudaHashDagGpuPlaceVoxelEditsDense(benchmark::State &state) {
    BM_CudaHashDagPlaceVoxelEdits<HashDagGpuBackend>(
        state, make_dense_edits<HashDagGpuBackend>);
}

void BM_CudaHashDagGpuPlaceVoxelEditsRandom(benchmark::State &state) {
    BM_CudaHashDagPlaceVoxelEdits<HashDagGpuBackend>(
        state, make_random_edits<HashDagGpuBackend>);
}

void BM_CudaHashDagGpuDestroyVoxelEditsDense(benchmark::State &state) {
    BM_CudaHashDagDestroyVoxelEdits<HashDagGpuBackend>(
        state, make_dense_edits<HashDagGpuBackend>);
}

void BM_CudaHashDagGpuDestroyVoxelEditsRandom(benchmark::State &state) {
    BM_CudaHashDagDestroyVoxelEdits<HashDagGpuBackend>(
        state, make_random_edits<HashDagGpuBackend>);
}

void BM_CudaHashDagGpuGetVoxelDense(benchmark::State &state) {
    BM_CudaHashDagGetVoxel<HashDagGpuBackend>(
        state, make_dense_edits<HashDagGpuBackend>);
}

void BM_CudaHashDagGpuGetVoxelRandom(benchmark::State &state) {
    BM_CudaHashDagGetVoxel<HashDagGpuBackend>(
        state, make_random_edits<HashDagGpuBackend>);
}

BENCHMARK(BM_CudaHashDagGpuPlaceVoxelEditsDense)
    ->UseManualTime()
    ->Apply(apply_hash_dag_args);
BENCHMARK(BM_CudaHashDagGpuPlaceVoxelEditsRandom)
    ->UseManualTime()
    ->Apply(apply_hash_dag_args);
BENCHMARK(BM_CudaHashDagGpuDestroyVoxelEditsDense)
    ->UseManualTime()
    ->Apply(apply_hash_dag_args);
BENCHMARK(BM_CudaHashDagGpuDestroyVoxelEditsRandom)
    ->UseManualTime()
    ->Apply(apply_hash_dag_args);
BENCHMARK(BM_CudaHashDagGpuGetVoxelDense)
    ->UseManualTime()
    ->Apply(apply_hash_dag_args);
BENCHMARK(BM_CudaHashDagGpuGetVoxelRandom)
    ->UseManualTime()
    ->Apply(apply_hash_dag_args);

} // namespace
