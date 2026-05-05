#include <algo/cuda/scan/scan.cuh>
#include <benchmark/benchmark.h>
#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using BlellochGlobal =
    algo::cuda::scan::Global<algo::cuda::scan::BlellochGlobal<256>>;
using BlellochShared256x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<256, 2>>>;
using BlellochSharedPadded256x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<
                256, 2, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochSharedPaddedReduceThenScan256x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ReduceThenScan<
            algo::cuda::scan::BlellochBlock<
                256, 2, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochSharedDecoupledLookback256x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::BlellochBlock<256, 2>>>;
using BlellochSharedPaddedDecoupledLookback256x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::BlellochBlock<
                256, 2, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochShared256x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<256, 4>>>;
using BlellochSharedPadded256x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<
                256, 4, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochSharedPaddedReduceThenScan256x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ReduceThenScan<
            algo::cuda::scan::BlellochBlock<
                256, 4, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochSharedDecoupledLookback256x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::BlellochBlock<256, 4>>>;
using BlellochSharedPaddedDecoupledLookback256x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::BlellochBlock<
                256, 4, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochShared512x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<512, 2>>>;
using BlellochSharedPadded512x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<
                512, 2, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochSharedPaddedReduceThenScan512x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ReduceThenScan<
            algo::cuda::scan::BlellochBlock<
                512, 2, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochShared512x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<512, 4>>>;
using BlellochSharedPadded512x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<
                512, 4, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochSharedPaddedReduceThenScan512x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ReduceThenScan<
            algo::cuda::scan::BlellochBlock<
                512, 4, algo::cuda::scan::PaddedSharedLayout>>>;
using HillisSteeleGlobal =
    algo::cuda::scan::Global<algo::cuda::scan::HillisSteeleGlobal<256>>;
using HillisSteeleShared256x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::HillisSteeleBlock<256, 2>>>;
using HillisSteeleSharedDecoupledLookback256x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::HillisSteeleBlock<256, 2>>>;
using HillisSteeleShared256x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::HillisSteeleBlock<256, 4>>>;
using HillisSteeleSharedDecoupledLookback256x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::HillisSteeleBlock<256, 4>>>;
using HillisSteeleShared512x2 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::HillisSteeleBlock<512, 2>>>;
using HillisSteeleShared512x4 =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::HillisSteeleBlock<512, 4>>>;
using WarpShuffle256x1 = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ScanThenPropagate<
        algo::cuda::scan::WarpShuffleBlock<256, 1>>>;
using WarpShuffleDecoupledLookback256x1 = algo::cuda::scan::BlockBased<
    algo::cuda::scan::DecoupledLookback<
        algo::cuda::scan::WarpShuffleBlock<256, 1>>>;
using WarpShuffle256x4 = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ScanThenPropagate<
        algo::cuda::scan::WarpShuffleBlock<256, 4>>>;
using WarpShuffle512x4 = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ScanThenPropagate<
        algo::cuda::scan::WarpShuffleBlock<512, 4>>>;
using WarpShuffleDecoupledLookback512x4 = algo::cuda::scan::BlockBased<
    algo::cuda::scan::DecoupledLookback<
        algo::cuda::scan::WarpShuffleBlock<512, 4>>>;
using WarpShuffleDecoupledLookback256x4 = algo::cuda::scan::BlockBased<
    algo::cuda::scan::DecoupledLookback<
        algo::cuda::scan::WarpShuffleBlock<256, 4>>>;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <class T>
void fill_random(std::vector<T>& values) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<T> dist(0, 9);
    for (auto& value : values) value = dist(rng);
}

template <class Config, class T>
void run_benchmark(benchmark::State& state, bool inclusive) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    std::vector<T> host_input(count);
    fill_random(host_input);

    T* d_input = nullptr;
    T* d_data = nullptr;
    std::byte* workspace = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    benchmark::DoNotOptimize(host_input.data());
    if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(T) * count) !=
        cudaSuccess) {
        state.SkipWithError("cudaMalloc failed for source buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(T) * count) !=
        cudaSuccess) {
        cudaFree(d_input);
        state.SkipWithError("cudaMalloc failed for input buffer");
        return;
    }

    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size<Config, T>(count);
    if (cudaMalloc(reinterpret_cast<void**>(&workspace),
                   workspace_size == 0 ? 1 : workspace_size) != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_data);
        state.SkipWithError("cudaMalloc failed for workspace");
        return;
    }

    if (cudaMemcpy(d_input, host_input.data(), sizeof(T) * count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(workspace);
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("cudaMemcpy failed for source buffer");
        return;
    }

    if (cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        cudaFree(workspace);
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    for (auto _ : state) {
        if (cudaMemcpy(d_data, d_input, sizeof(T) * count,
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            state.SkipWithError("cudaMemcpyDeviceToDevice failed");
            break;
        }

        if (cudaEventRecord(start) != cudaSuccess) {
            state.SkipWithError("cudaEventRecord(start) failed");
            break;
        }
        if (inclusive) {
            if ((algo::cuda::scan::inclusive_sum<Config>(
                     d_data, count, workspace, workspace_size)) !=
                cudaSuccess) {
                state.SkipWithError("inclusive_sum failed");
                break;
            }
        } else {
            if ((algo::cuda::scan::exclusive_sum<Config>(
                     d_data, count, workspace, workspace_size)) !=
                cudaSuccess) {
                state.SkipWithError("exclusive_sum failed");
                break;
            }
        }

        if (cudaEventRecord(stop) != cudaSuccess) {
            state.SkipWithError("cudaEventRecord(stop) failed");
            break;
        }
        if (cudaEventSynchronize(stop) != cudaSuccess) {
            state.SkipWithError("cudaEventSynchronize failed");
            break;
        }

        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) {
            state.SkipWithError("cudaEventElapsedTime failed");
            break;
        }
        state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count) *
                            static_cast<int64_t>(sizeof(T)));

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(workspace);
    cudaFree(d_data);
    cudaFree(d_input);
}

template <class T>
void run_cub_benchmark(benchmark::State& state, bool inclusive) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    std::vector<T> host_input(count);
    fill_random(host_input);

    T* d_input = nullptr;
    T* d_data = nullptr;
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    benchmark::DoNotOptimize(host_input.data());
    if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(T) * count) !=
        cudaSuccess) {
        state.SkipWithError("cudaMalloc failed for source buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(T) * count) !=
        cudaSuccess) {
        cudaFree(d_input);
        state.SkipWithError("cudaMalloc failed for input buffer");
        return;
    }

    if (cudaMemcpy(d_input, host_input.data(), sizeof(T) * count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("cudaMemcpy failed for source buffer");
        return;
    }

    const cudaError_t storage_status =
        inclusive ? cub::DeviceScan::InclusiveSum(d_temp_storage,
                                                  temp_storage_bytes, d_data,
                                                  d_data, count)
                  : cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                                  temp_storage_bytes, d_data,
                                                  d_data, count);
    if (storage_status != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("CUB temp storage size query failed");
        return;
    }
    if (cudaMalloc(&d_temp_storage, temp_storage_bytes) != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("cudaMalloc failed for CUB temp storage");
        return;
    }

    if (cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        cudaFree(d_temp_storage);
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    for (auto _ : state) {
        if (cudaMemcpy(d_data, d_input, sizeof(T) * count,
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            state.SkipWithError("cudaMemcpyDeviceToDevice failed");
            break;
        }

        if (cudaEventRecord(start) != cudaSuccess) {
            state.SkipWithError("cudaEventRecord(start) failed");
            break;
        }

        const cudaError_t status =
            inclusive ? cub::DeviceScan::InclusiveSum(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_data, d_data, count)
                      : cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_data, d_data, count);
        if (status != cudaSuccess) {
            state.SkipWithError("CUB scan failed");
            break;
        }

        if (cudaEventRecord(stop) != cudaSuccess) {
            state.SkipWithError("cudaEventRecord(stop) failed");
            break;
        }
        if (cudaEventSynchronize(stop) != cudaSuccess) {
            state.SkipWithError("cudaEventSynchronize failed");
            break;
        }

        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) {
            state.SkipWithError("cudaEventElapsedTime failed");
            break;
        }
        state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count) *
                            static_cast<int64_t>(sizeof(T)));

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_temp_storage);
    cudaFree(d_data);
    cudaFree(d_input);
}

void apply_scan_args(benchmark::internal::Benchmark* bench) {
    bench->Arg(1 << 18);
    bench->Arg(1 << 20);
    bench->Arg(1 << 22);
}

#define SCAN_BENCHMARKS(X)                                                  \
    X(BL_G_256x1, BlellochGlobal)                                           \
    X(HS_G_256x1, HillisSteeleGlobal)                                       \
                                                                            \
    X(BL_S_256x2, BlellochShared256x2)                                      \
    X(BL_SP_256x2, BlellochSharedPadded256x2)                               \
    X(BL_S_256x4, BlellochShared256x4)                                      \
    X(BL_SP_256x4, BlellochSharedPadded256x4)                               \
    X(BL_S_512x2, BlellochShared512x2)                                      \
    X(BL_SP_512x2, BlellochSharedPadded512x2)                               \
    X(BL_S_512x4, BlellochShared512x4)                                      \
    X(BL_SP_512x4, BlellochSharedPadded512x4)                               \
                                                                            \
    X(HS_S_256x2, HillisSteeleShared256x2)                                  \
    X(HS_S_256x4, HillisSteeleShared256x4)                                  \
    X(HS_S_512x2, HillisSteeleShared512x2)                                  \
    X(HS_S_512x4, HillisSteeleShared512x4)                                  \
                                                                            \
    X(WS_256x1, WarpShuffle256x1)                                           \
    X(WS_256x4, WarpShuffle256x4)                                           \
    X(WS_512x4, WarpShuffle512x4)                                           \
                                                                            \
    X(BL_RTSP_256x2, BlellochSharedPaddedReduceThenScan256x2)               \
    X(BL_RTSP_256x4, BlellochSharedPaddedReduceThenScan256x4)               \
    X(BL_RTSP_512x2, BlellochSharedPaddedReduceThenScan512x2)               \
    X(BL_RTSP_512x4, BlellochSharedPaddedReduceThenScan512x4)               \
                                                                            \
    X(BL_DLB_256x2, BlellochSharedDecoupledLookback256x2)                   \
    X(BL_DLB_256x4, BlellochSharedDecoupledLookback256x4)                   \
    X(BL_SP_DLB_256x2, BlellochSharedPaddedDecoupledLookback256x2)          \
    X(BL_SP_DLB_256x4, BlellochSharedPaddedDecoupledLookback256x4)          \
    X(HS_DLB_256x2, HillisSteeleSharedDecoupledLookback256x2)               \
    X(HS_DLB_256x4, HillisSteeleSharedDecoupledLookback256x4)               \
    X(WS_DLB_256x1, WarpShuffleDecoupledLookback256x1)                      \
    X(WS_DLB_256x4, WarpShuffleDecoupledLookback256x4)                      \
    X(WS_DLB_512x4, WarpShuffleDecoupledLookback512x4)

#define DEFINE_SCAN_BENCHMARK(Name, Config)                                 \
    void BM_##Name##_In(benchmark::State& state) {                          \
        run_benchmark<Config, std::uint32_t>(state, true);                  \
    }                                                                       \
                                                                            \
    void BM_##Name##_Ex(benchmark::State& state) {                          \
        run_benchmark<Config, std::uint32_t>(state, false);                 \
    }

SCAN_BENCHMARKS(DEFINE_SCAN_BENCHMARK)

#define REGISTER_SCAN_BENCHMARK(Name, Config)                               \
    BENCHMARK(BM_##Name##_In)->UseManualTime()->Apply(apply_scan_args);     \
    BENCHMARK(BM_##Name##_Ex)->UseManualTime()->Apply(apply_scan_args);

SCAN_BENCHMARKS(REGISTER_SCAN_BENCHMARK)

void BM_CUB_In(benchmark::State& state) {
    run_cub_benchmark<std::uint32_t>(state, true);
}

void BM_CUB_Ex(benchmark::State& state) {
    run_cub_benchmark<std::uint32_t>(state, false);
}

BENCHMARK(BM_CUB_In)->UseManualTime()->Apply(apply_scan_args);
BENCHMARK(BM_CUB_Ex)->UseManualTime()->Apply(apply_scan_args);

#undef REGISTER_SCAN_BENCHMARK
#undef DEFINE_SCAN_BENCHMARK
#undef SCAN_BENCHMARKS

} // namespace
