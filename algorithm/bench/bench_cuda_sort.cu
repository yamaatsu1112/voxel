#include <algo/cuda/sort/sort.cuh>
#include <benchmark/benchmark.h>
#include <cub/device/device_radix_sort.cuh>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct Value32x4 {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t w;
};

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <class T>
void fill_random(std::vector<T>& values) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<T> dist(0, 0xffffu);
    for (auto& value : values) value = dist(rng);
}

template <class T>
void fill_values(std::vector<T>& values) {
    std::mt19937 rng(456);
    std::uniform_int_distribution<std::uint32_t> dist(0, 0xffffu);
    for (auto& value : values) value = static_cast<T>(dist(rng));
}

template <>
void fill_values(std::vector<Value32x4>& values) {
    std::mt19937 rng(456);
    std::uniform_int_distribution<std::uint32_t> dist(0, 0xffffu);
    for (auto& value : values) {
        value = Value32x4{dist(rng), dist(rng), dist(rng), dist(rng)};
    }
}

template <class Config, class T>
void run_benchmark(benchmark::State& state) {
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
        algo::cuda::sort::required_workspace_size<Config, T>(count);
    if (cudaMalloc(reinterpret_cast<void**>(&workspace),
                   workspace_size == 0 ? 1 : workspace_size) != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_input);
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

        if ((algo::cuda::sort::sort_keys<Config>(
                d_data, count, workspace, workspace_size)) != cudaSuccess) {
            state.SkipWithError("sort_keys failed");
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
    cudaFree(workspace);
    cudaFree(d_data);
    cudaFree(d_input);
}

template <class Config, class Value>
void run_by_key_benchmark(benchmark::State& state) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    std::vector<std::uint32_t> host_keys(count);
    std::vector<Value> host_values(count);
    fill_random(host_keys);
    fill_values(host_values);

    std::uint32_t* d_input_keys = nullptr;
    std::uint32_t* d_keys = nullptr;
    Value* d_input_values = nullptr;
    Value* d_values = nullptr;
    std::byte* workspace = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    benchmark::DoNotOptimize(host_keys.data());
    benchmark::DoNotOptimize(host_values.data());

    if (cudaMalloc(reinterpret_cast<void**>(&d_input_keys),
                   sizeof(std::uint32_t) * count) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed for source key buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_keys),
                   sizeof(std::uint32_t) * count) != cudaSuccess) {
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for key buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_input_values),
                   sizeof(Value) * count) != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for source value buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_values), sizeof(Value) * count) !=
        cudaSuccess) {
        cudaFree(d_input_values);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for value buffer");
        return;
    }

    const std::size_t workspace_size =
        algo::cuda::sort::required_sort_by_key_workspace_size<
            Config, std::uint32_t, Value>(count);
    if (cudaMalloc(reinterpret_cast<void**>(&workspace),
                   workspace_size == 0 ? 1 : workspace_size) != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for workspace");
        return;
    }

    if (cudaMemcpy(d_input_keys, host_keys.data(),
                   sizeof(std::uint32_t) * count, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        cudaFree(workspace);
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMemcpy failed for source key buffer");
        return;
    }
    if (cudaMemcpy(d_input_values, host_values.data(), sizeof(Value) * count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(workspace);
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMemcpy failed for source value buffer");
        return;
    }

    if (cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        cudaFree(workspace);
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    for (auto _ : state) {
        if (cudaMemcpy(d_keys, d_input_keys, sizeof(std::uint32_t) * count,
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            state.SkipWithError("cudaMemcpyDeviceToDevice failed for keys");
            break;
        }
        if (cudaMemcpy(d_values, d_input_values, sizeof(Value) * count,
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            state.SkipWithError("cudaMemcpyDeviceToDevice failed for values");
            break;
        }

        if (cudaEventRecord(start) != cudaSuccess) {
            state.SkipWithError("cudaEventRecord(start) failed");
            break;
        }

        if ((algo::cuda::sort::sort_by_key<Config>(
                d_keys, algo::cuda::sort::value_arrays(d_values), count,
                workspace, workspace_size)) !=
            cudaSuccess) {
            state.SkipWithError("sort_by_key failed");
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
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(count) *
        static_cast<int64_t>(sizeof(std::uint32_t) + sizeof(Value)));

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(workspace);
    cudaFree(d_values);
    cudaFree(d_input_values);
    cudaFree(d_keys);
    cudaFree(d_input_keys);
}

template <class Config>
void run_by_key_value_arrays32x4_benchmark(benchmark::State& state) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    std::vector<std::uint32_t> host_keys(count);
    std::vector<std::uint32_t> host_x(count);
    std::vector<std::uint32_t> host_y(count);
    std::vector<std::uint32_t> host_z(count);
    std::vector<std::uint32_t> host_w(count);
    fill_random(host_keys);
    fill_values(host_x);
    fill_values(host_y);
    fill_values(host_z);
    fill_values(host_w);

    std::uint32_t* d_input_keys = nullptr;
    std::uint32_t* d_keys = nullptr;
    std::uint32_t* d_input_x = nullptr;
    std::uint32_t* d_input_y = nullptr;
    std::uint32_t* d_input_z = nullptr;
    std::uint32_t* d_input_w = nullptr;
    std::uint32_t* d_x = nullptr;
    std::uint32_t* d_y = nullptr;
    std::uint32_t* d_z = nullptr;
    std::uint32_t* d_w = nullptr;
    std::byte* workspace = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    benchmark::DoNotOptimize(host_keys.data());
    benchmark::DoNotOptimize(host_x.data());
    benchmark::DoNotOptimize(host_y.data());
    benchmark::DoNotOptimize(host_z.data());
    benchmark::DoNotOptimize(host_w.data());

    auto cleanup = [&]() {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        cudaFree(workspace);
        cudaFree(d_w);
        cudaFree(d_z);
        cudaFree(d_y);
        cudaFree(d_x);
        cudaFree(d_input_w);
        cudaFree(d_input_z);
        cudaFree(d_input_y);
        cudaFree(d_input_x);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
    };

    auto allocate_u32 = [&](std::uint32_t** ptr, const char* message) {
        if (cudaMalloc(reinterpret_cast<void**>(ptr),
                       sizeof(std::uint32_t) * count) != cudaSuccess) {
            cleanup();
            state.SkipWithError(message);
            return false;
        }
        return true;
    };

    if (!allocate_u32(&d_input_keys,
                      "cudaMalloc failed for source key buffer") ||
        !allocate_u32(&d_keys, "cudaMalloc failed for key buffer") ||
        !allocate_u32(&d_input_x, "cudaMalloc failed for source x buffer") ||
        !allocate_u32(&d_input_y, "cudaMalloc failed for source y buffer") ||
        !allocate_u32(&d_input_z, "cudaMalloc failed for source z buffer") ||
        !allocate_u32(&d_input_w, "cudaMalloc failed for source w buffer") ||
        !allocate_u32(&d_x, "cudaMalloc failed for x buffer") ||
        !allocate_u32(&d_y, "cudaMalloc failed for y buffer") ||
        !allocate_u32(&d_z, "cudaMalloc failed for z buffer") ||
        !allocate_u32(&d_w, "cudaMalloc failed for w buffer")) {
        return;
    }

    const std::size_t workspace_size =
        algo::cuda::sort::required_sort_by_key_workspace_size<
            Config, std::uint32_t, std::uint32_t, std::uint32_t,
            std::uint32_t, std::uint32_t>(count);
    if (cudaMalloc(reinterpret_cast<void**>(&workspace),
                   workspace_size == 0 ? 1 : workspace_size) != cudaSuccess) {
        cleanup();
        state.SkipWithError("cudaMalloc failed for workspace");
        return;
    }

    auto copy_to_device = [&](std::uint32_t* dst,
                              const std::vector<std::uint32_t>& src,
                              const char* message) {
        if (cudaMemcpy(dst, src.data(), sizeof(std::uint32_t) * count,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cleanup();
            state.SkipWithError(message);
            return false;
        }
        return true;
    };

    if (!copy_to_device(d_input_keys, host_keys,
                        "cudaMemcpy failed for source key buffer") ||
        !copy_to_device(d_input_x, host_x,
                        "cudaMemcpy failed for source x buffer") ||
        !copy_to_device(d_input_y, host_y,
                        "cudaMemcpy failed for source y buffer") ||
        !copy_to_device(d_input_z, host_z,
                        "cudaMemcpy failed for source z buffer") ||
        !copy_to_device(d_input_w, host_w,
                        "cudaMemcpy failed for source w buffer")) {
        return;
    }

    if (cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) {
        cleanup();
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    auto copy_device = [&](std::uint32_t* dst, const std::uint32_t* src,
                           const char* message) {
        if (cudaMemcpy(dst, src, sizeof(std::uint32_t) * count,
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            state.SkipWithError(message);
            return false;
        }
        return true;
    };

    for (auto _ : state) {
        if (!copy_device(d_keys, d_input_keys,
                         "cudaMemcpyDeviceToDevice failed for keys") ||
            !copy_device(d_x, d_input_x,
                         "cudaMemcpyDeviceToDevice failed for x") ||
            !copy_device(d_y, d_input_y,
                         "cudaMemcpyDeviceToDevice failed for y") ||
            !copy_device(d_z, d_input_z,
                         "cudaMemcpyDeviceToDevice failed for z") ||
            !copy_device(d_w, d_input_w,
                         "cudaMemcpyDeviceToDevice failed for w")) {
            break;
        }

        if (cudaEventRecord(start) != cudaSuccess) {
            state.SkipWithError("cudaEventRecord(start) failed");
            break;
        }

        if ((algo::cuda::sort::sort_by_key<Config>(
                d_keys, algo::cuda::sort::value_arrays(d_x, d_y, d_z, d_w),
                count, workspace, workspace_size)) != cudaSuccess) {
            state.SkipWithError("sort_by_key ValueArrays32x4 failed");
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
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(count) *
        static_cast<int64_t>(sizeof(std::uint32_t) * 5u));

    cleanup();
}

template <class T>
void run_cub_benchmark(benchmark::State& state) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    std::vector<T> host_input(count);
    fill_random(host_input);

    T* d_input = nullptr;
    T* d_data = nullptr;
    T* d_output = nullptr;
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
    if (cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(T) * count) !=
        cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("cudaMalloc failed for output buffer");
        return;
    }

    if (cudaMemcpy(d_input, host_input.data(), sizeof(T) * count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_output);
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("cudaMemcpy failed for source buffer");
        return;
    }

    const cudaError_t storage_status = cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, d_data, d_output, count);
    if (storage_status != cudaSuccess) {
        cudaFree(d_output);
        cudaFree(d_data);
        cudaFree(d_input);
        state.SkipWithError("CUB temp storage size query failed");
        return;
    }
    if (cudaMalloc(&d_temp_storage,
                   temp_storage_bytes == 0 ? 1 : temp_storage_bytes) !=
        cudaSuccess) {
        cudaFree(d_output);
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
        cudaFree(d_output);
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

        const cudaError_t status = cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_data, d_output, count);
        if (status != cudaSuccess) {
            state.SkipWithError("CUB sort failed");
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
    cudaFree(d_output);
    cudaFree(d_data);
    cudaFree(d_input);
}

template <class Value>
void run_cub_pairs_benchmark(benchmark::State& state) {
    if (!has_cuda_device()) {
        state.SkipWithError("CUDA device is not available");
        return;
    }

    const auto count = static_cast<std::uint32_t>(state.range(0));
    std::vector<std::uint32_t> host_keys(count);
    std::vector<Value> host_values(count);
    fill_random(host_keys);
    fill_values(host_values);

    std::uint32_t* d_input_keys = nullptr;
    std::uint32_t* d_keys = nullptr;
    std::uint32_t* d_output_keys = nullptr;
    Value* d_input_values = nullptr;
    Value* d_values = nullptr;
    Value* d_output_values = nullptr;
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    benchmark::DoNotOptimize(host_keys.data());
    benchmark::DoNotOptimize(host_values.data());

    if (cudaMalloc(reinterpret_cast<void**>(&d_input_keys),
                   sizeof(std::uint32_t) * count) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed for source key buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_keys),
                   sizeof(std::uint32_t) * count) != cudaSuccess) {
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for key buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_output_keys),
                   sizeof(std::uint32_t) * count) != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for output key buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_input_values),
                   sizeof(Value) * count) != cudaSuccess) {
        cudaFree(d_output_keys);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for source value buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_values), sizeof(Value) * count) !=
        cudaSuccess) {
        cudaFree(d_input_values);
        cudaFree(d_output_keys);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for value buffer");
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_output_values),
                   sizeof(Value) * count) != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_output_keys);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for output value buffer");
        return;
    }

    if (cudaMemcpy(d_input_keys, host_keys.data(),
                   sizeof(std::uint32_t) * count, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        cudaFree(d_output_values);
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_output_keys);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMemcpy failed for source key buffer");
        return;
    }
    if (cudaMemcpy(d_input_values, host_values.data(), sizeof(Value) * count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_output_values);
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_output_keys);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMemcpy failed for source value buffer");
        return;
    }

    const cudaError_t storage_status = cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_keys, d_output_keys, d_values,
        d_output_values, count);
    if (storage_status != cudaSuccess) {
        cudaFree(d_output_values);
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_output_keys);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("CUB temp storage size query failed");
        return;
    }
    if (cudaMalloc(&d_temp_storage,
                   temp_storage_bytes == 0 ? 1 : temp_storage_bytes) !=
        cudaSuccess) {
        cudaFree(d_output_values);
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_output_keys);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaMalloc failed for CUB temp storage");
        return;
    }

    if (cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        cudaFree(d_temp_storage);
        cudaFree(d_output_values);
        cudaFree(d_values);
        cudaFree(d_input_values);
        cudaFree(d_output_keys);
        cudaFree(d_keys);
        cudaFree(d_input_keys);
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    for (auto _ : state) {
        if (cudaMemcpy(d_keys, d_input_keys, sizeof(std::uint32_t) * count,
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            state.SkipWithError("cudaMemcpyDeviceToDevice failed for keys");
            break;
        }
        if (cudaMemcpy(d_values, d_input_values, sizeof(Value) * count,
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            state.SkipWithError("cudaMemcpyDeviceToDevice failed for values");
            break;
        }

        if (cudaEventRecord(start) != cudaSuccess) {
            state.SkipWithError("cudaEventRecord(start) failed");
            break;
        }

        const cudaError_t status = cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_keys, d_output_keys, d_values,
            d_output_values, count);
        if (status != cudaSuccess) {
            state.SkipWithError("CUB sort_by_key failed");
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
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(count) *
        static_cast<int64_t>(sizeof(std::uint32_t) + sizeof(Value)));

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_temp_storage);
    cudaFree(d_output_values);
    cudaFree(d_values);
    cudaFree(d_input_values);
    cudaFree(d_output_keys);
    cudaFree(d_keys);
    cudaFree(d_input_keys);
}

void apply_sort_args(benchmark::internal::Benchmark* bench) {
    bench->Arg(1 << 11)
        ->Arg(1 << 13)
        ->Arg(1 << 15)
        ->Arg(1 << 17)
        ->Arg(1 << 19)
        ->Arg(1 << 20)
        ->Arg(1 << 22);
}

#define SORT_BENCHMARKS(X)                                                     \
    X(RS_256_2BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::FlagPrefixSumPass<>, 256, 2>) \
    X(RS_256_4BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::FlagPrefixSumPass<>, 256, 4>) \
    X(RS_FLAT_256_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::FlagPrefixSumPass<algo::cuda::sort::FlattenedScan>, 256, 2>) \
    X(RS_FLAT_256_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::FlagPrefixSumPass<algo::cuda::sort::FlattenedScan>, 256, 4>) \
    X(RS_512_2BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::FlagPrefixSumPass<>, 512, 2>) \
    X(RS_512_4BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::FlagPrefixSumPass<>, 512, 4>) \
    X(RS_FLAT_512_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::FlagPrefixSumPass<algo::cuda::sort::FlattenedScan>, 512, 2>) \
    X(RS_FLAT_512_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::FlagPrefixSumPass<algo::cuda::sort::FlattenedScan>, 512, 4>) \
    X(RS_HIST_SHARED_ATOMIC_BUCKET_BALLOT_256_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 256, 2>) \
    X(RS_HIST_SHARED_ATOMIC_BUCKET_BALLOT_256_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 256, 4>) \
    X(RS_HIST_SHARED_ATOMIC_BUCKET_BALLOT_256_8BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 256, 8>) \
    X(RS_HIST_SHARED_ATOMIC_BUCKET_BALLOT_512_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 512, 2>) \
    X(RS_HIST_SHARED_ATOMIC_BUCKET_BALLOT_512_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 512, 4>) \
    X(RS_HIST_SHARED_ATOMIC_BUCKET_BALLOT_512_8BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 512, 8>) \
    X(RS_HIST_WARP_BALLOT_BUCKET_BALLOT_256_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpBallotHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 256, 2>) \
    X(RS_HIST_WARP_BALLOT_BUCKET_BALLOT_256_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpBallotHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 256, 4>) \
    X(RS_HIST_WARP_BALLOT_BUCKET_BALLOT_256_8BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpBallotHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 256, 8>) \
    X(RS_HIST_WARP_BALLOT_BUCKET_BALLOT_512_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpBallotHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 512, 2>) \
    X(RS_HIST_WARP_BALLOT_BUCKET_BALLOT_512_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpBallotHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 512, 4>) \
    X(RS_HIST_WARP_BALLOT_BUCKET_BALLOT_512_8BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpBallotHistogram, algo::cuda::sort::BucketBallotWarpRank, 1>, 512, 8>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_256_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 256, 2>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_256_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 256, 4>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_256_8BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 256, 8>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 512, 2>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 512, 4>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_8BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 512, 8>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_2BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>, 512, 2>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_4BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>, 512, 4>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_8BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>, 512, 8>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_2BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 8>, 512, 2>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_4BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 8>, 512, 4>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_8BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 8>, 512, 8>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_2BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 16>, 512, 2>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_4BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 16>, 512, 4>) \
    X(RS_HIST_WARP_LEVEL_MULTI_SPLIT_512_8BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::WarpLevelMultiSplitHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 16>, 512, 8>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_256_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 256, 2>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_256_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 256, 4>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_256_8BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 256, 8>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_2BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 512, 2>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_4BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 512, 4>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_8BITS, algo::cuda::sort::RadixSort< algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 1>, 512, 8>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_2BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>, 512, 2>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_4BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>, 512, 4>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_8BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>, 512, 8>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_2BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 8>, 512, 2>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_4BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 8>, 512, 4>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_8BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 8>, 512, 8>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_2BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 16>, 512, 2>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_4BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 16>, 512, 4>) \
    X(RS_HIST_SHARED_ATOMIC_WARP_LEVEL_MULTI_SPLIT_512_8BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::HistogramPass<algo::cuda::sort::SharedAtomicHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, 16>, 512, 8>) \
    X(RS_ONESWEEP_256_2BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 1>, 256, 2>) \
    X(RS_ONESWEEP_256_4BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 1>, 256, 4>) \
    X(RS_ONESWEEP_256_8BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 1>, 256, 8>) \
    X(RS_ONESWEEP_256_2BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 4>, 256, 2>) \
    X(RS_ONESWEEP_256_4BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 4>, 256, 4>) \
    X(RS_ONESWEEP_256_8BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 4>, 256, 8>) \
    X(RS_ONESWEEP_256_2BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 8>, 256, 2>) \
    X(RS_ONESWEEP_256_4BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 8>, 256, 4>) \
    X(RS_ONESWEEP_256_8BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 8>, 256, 8>) \
    X(RS_ONESWEEP_256_2BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 16>, 256, 2>) \
    X(RS_ONESWEEP_256_4BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 16>, 256, 4>) \
    X(RS_ONESWEEP_256_8BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 16>, 256, 8>) \
    X(RS_ONESWEEP_512_2BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 1>, 512, 2>) \
    X(RS_ONESWEEP_512_4BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 1>, 512, 4>) \
    X(RS_ONESWEEP_512_8BITS, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 1>, 512, 8>) \
    X(RS_ONESWEEP_512_2BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 4>, 512, 2>) \
    X(RS_ONESWEEP_512_4BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 4>, 512, 4>) \
    X(RS_ONESWEEP_512_8BITS_ITEMS4, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 4>, 512, 8>) \
    X(RS_ONESWEEP_512_2BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 8>, 512, 2>) \
    X(RS_ONESWEEP_512_4BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 8>, 512, 4>) \
    X(RS_ONESWEEP_512_8BITS_ITEMS8, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 8>, 512, 8>) \
    X(RS_ONESWEEP_512_2BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 16>, 512, 2>) \
    X(RS_ONESWEEP_512_4BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 16>, 512, 4>) \
    X(RS_ONESWEEP_512_8BITS_ITEMS16, algo::cuda::sort::RadixSort<algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram, algo::cuda::sort::WarpLevelMultiSplitWarpRank, algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram, 16>, 512, 8>) \


#define DEFINE_SORT_BENCHMARK(Name, ...)                                      \
    void BM_##Name(benchmark::State& state) {                                  \
        run_benchmark<__VA_ARGS__, std::uint32_t>(state);                      \
    }

#define DEFINE_SORT_BY_KEY_BENCHMARK(Name, ...)                                 \
    void BM_##Name##_P32(benchmark::State& state) {                            \
        run_by_key_benchmark<__VA_ARGS__, std::uint32_t>(state);                \
    }                                                                          \
                                                                               \
    void BM_##Name##_P32x4(benchmark::State& state) {                          \
        run_by_key_benchmark<__VA_ARGS__, Value32x4>(state);                    \
    }                                                                          \
                                                                               \
    void BM_##Name##_VALUE_ARRAYS32x4(benchmark::State& state) {                         \
        run_by_key_value_arrays32x4_benchmark<__VA_ARGS__>(state);                       \
    }

SORT_BENCHMARKS(DEFINE_SORT_BENCHMARK)
SORT_BENCHMARKS(DEFINE_SORT_BY_KEY_BENCHMARK)

void BM_CUB(benchmark::State& state) {
    run_cub_benchmark<std::uint32_t>(state);
}

void BM_CUB_P32(benchmark::State& state) {
    run_cub_pairs_benchmark<std::uint32_t>(state);
}

void BM_CUB_P32x4(benchmark::State& state) {
    run_cub_pairs_benchmark<Value32x4>(state);
}

#define REGISTER_SORT_BENCHMARK(Name, ...)                                  \
    BENCHMARK(BM_##Name)->UseManualTime()->Apply(apply_sort_args);

#define REGISTER_SORT_PAIR_BENCHMARK(Name, ...)                             \
    BENCHMARK(BM_##Name##_P32)->UseManualTime()->Apply(apply_sort_args);       \
    BENCHMARK(BM_##Name##_P32x4)->UseManualTime()->Apply(apply_sort_args);      \
    BENCHMARK(BM_##Name##_VALUE_ARRAYS32x4)->UseManualTime()->Apply(apply_sort_args);

SORT_BENCHMARKS(REGISTER_SORT_BENCHMARK)
BENCHMARK(BM_CUB)->UseManualTime()->Apply(apply_sort_args);
SORT_BENCHMARKS(REGISTER_SORT_PAIR_BENCHMARK)

BENCHMARK(BM_CUB_P32)->UseManualTime()->Apply(apply_sort_args);
BENCHMARK(BM_CUB_P32x4)->UseManualTime()->Apply(apply_sort_args);

#undef REGISTER_SORT_PAIR_BENCHMARK
#undef REGISTER_SORT_BENCHMARK
#undef DEFINE_SORT_BY_KEY_BENCHMARK
#undef DEFINE_SORT_BENCHMARK
#undef SORT_BENCHMARKS

} // namespace
