#include <algo/cuda/acceleration_hash.cuh>
#include <algo/cuda/hash_table/bump_allocator.cuh>
#include <algo/cuda/slab_hash.cuh>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <vector>

namespace {

constexpr int kRangeMin = 1 << 10;
constexpr int kRangeMax = 1 << 20;
constexpr int kRangeMultiplier = 4;

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::uint32_t make_key(std::uint32_t index) {
  std::uint32_t key = index * 2654435761u + 1u;
  while (algo::cuda::slab_hash_is_reserved_key(key))
    key += 0x9e3779b9u;
  return key;
}

template <std::uint32_t ItemWords>
void fill_items(std::vector<std::uint32_t> &items, std::uint32_t count,
                std::uint32_t offset = 0) {
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint32_t base = make_key(i + offset);
    for (std::uint32_t word = 0; word < ItemWords; ++word) {
      items[i * ItemWords + word] = base ^ (word * 0x9e3779b9u) ^ (word << 16u);
    }
  }
}

std::uint32_t bucket_count_for(std::uint32_t count) {
  constexpr std::uint32_t kTargetEntriesPerBucket = 24;
  return std::max<std::uint32_t>(1, count / kTargetEntriesPerBucket);
}

std::uint32_t overflow_slab_capacity_for(std::uint32_t count,
                                         std::uint32_t bucket_count,
                                         std::uint32_t slots_per_slab) {
  const std::uint32_t total_needed =
      (count + slots_per_slab - 1u) / slots_per_slab;
  return std::max<std::uint32_t>(total_needed + bucket_count, 8);
}

template <class T> class DeviceAllocation {
public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  ~DeviceAllocation() {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
  }

  bool allocate(std::uint32_t count, benchmark::State &state,
                const char *label) {
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count);
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
  const cudaError_t status = cudaMemcpy(dst, src.data(), sizeof(T) * src.size(),
                                        cudaMemcpyHostToDevice);
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

class CudaEvents {
public:
  bool create(benchmark::State &state) {
    return create_events(&start_, &stop_, state);
  }

  [[nodiscard]] cudaEvent_t start() const { return start_; }
  [[nodiscard]] cudaEvent_t stop() const { return stop_; }

  ~CudaEvents() {
    if (stop_ != nullptr)
      cudaEventDestroy(stop_);
    if (start_ != nullptr)
      cudaEventDestroy(start_);
  }

private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

struct SlabHashSetBenchmark {
  using Table = algo::cuda::SlabHashSetU32;
  static constexpr std::uint32_t kItemWords = 1;
  static constexpr std::uint32_t kSlotsPerSlab =
      algo::cuda::kSlabHashSetKeysPerSlab;

  static cudaError_t insert(Table &table, const std::uint32_t *items,
                            std::uint32_t count, std::uint32_t *results,
                            std::uint32_t *statuses) {
    (void)results;
    return table.insert_batch(items, count, statuses);
  }

  static cudaError_t contains(const Table &table, const std::uint32_t *items,
                              std::uint32_t count, std::uint32_t *results,
                              std::uint32_t *statuses) {
    return table.contains_batch(items, results, count, statuses);
  }
};

struct BumpSlabHashSetBenchmark {
  using Table = algo::cuda::BasicSlabHashSetU32<algo::cuda::BumpSlabAllocator>;
  static constexpr std::uint32_t kItemWords = 1;
  static constexpr std::uint32_t kSlotsPerSlab =
      algo::cuda::kSlabHashSetKeysPerSlab;

  static cudaError_t insert(Table &table, const std::uint32_t *items,
                            std::uint32_t count, std::uint32_t *results,
                            std::uint32_t *statuses) {
    (void)results;
    return table.insert_batch(items, count, statuses);
  }

  static cudaError_t contains(const Table &table, const std::uint32_t *items,
                              std::uint32_t count, std::uint32_t *results,
                              std::uint32_t *statuses) {
    return table.contains_batch(items, results, count, statuses);
  }
};

struct AccelerationHashSetBenchmark {
  using Table = algo::cuda::AccelerationHashSet32<1>;
  static constexpr std::uint32_t kItemWords = 1;
  static constexpr std::uint32_t kSlotsPerSlab =
      algo::cuda::kAccelerationHashSlotsPerSlab;

  static cudaError_t insert(Table &table, const std::uint32_t *items,
                            std::uint32_t count, std::uint32_t *results,
                            std::uint32_t *statuses) {
    return table.insert_unique_unchecked_batch(items, count, results, statuses);
  }

  static cudaError_t contains(const Table &table, const std::uint32_t *items,
                              std::uint32_t count, std::uint32_t *results,
                              std::uint32_t *statuses) {
    return table.find_batch(items, count, results, statuses);
  }
};

template <class HashSet> void BM_HashSetInsert(benchmark::State &state) {
  if (!has_cuda_device()) {
    state.SkipWithError("CUDA device is not available");
    return;
  }

  const auto count = static_cast<std::uint32_t>(state.range(0));
  std::vector<std::uint32_t> host_items(count * HashSet::kItemWords);
  fill_items<HashSet::kItemWords>(host_items, count);

  DeviceAllocation<std::uint32_t> items;
  DeviceAllocation<std::uint32_t> results;
  DeviceAllocation<std::uint32_t> statuses;
  CudaEvents events;
  if (!items.allocate(count * HashSet::kItemWords, state,
                      "cudaMalloc failed for items") ||
      !results.allocate(count, state, "cudaMalloc failed for results") ||
      !statuses.allocate(count, state, "cudaMalloc failed for statuses") ||
      !copy_to_device(items.get(), host_items, state,
                      "cudaMemcpy failed for items") ||
      !events.create(state)) {
    return;
  }

  const std::uint32_t buckets = bucket_count_for(count);
  try {
    typename HashSet::Table table(
        buckets,
        overflow_slab_capacity_for(count, buckets, HashSet::kSlotsPerSlab));

    for (auto _ : state) {
      table.clear();
      if (cudaEventRecord(events.start()) != cudaSuccess) {
        state.SkipWithError("cudaEventRecord(start) failed");
        break;
      }
      if (HashSet::insert(table, items.get(), count, results.get(),
                          statuses.get()) != cudaSuccess) {
        state.SkipWithError("hash set insert batch failed");
        break;
      }
      if (!record_elapsed(events.start(), events.stop(), state))
        break;
    }
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(count));
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(count) *
      static_cast<int64_t>(sizeof(std::uint32_t) * (HashSet::kItemWords + 1)));
}

template <class HashSet>
void BM_HashSetContains(benchmark::State &state, bool successful) {
  if (!has_cuda_device()) {
    state.SkipWithError("CUDA device is not available");
    return;
  }

  const auto count = static_cast<std::uint32_t>(state.range(0));
  std::vector<std::uint32_t> host_insert_items(count * HashSet::kItemWords);
  std::vector<std::uint32_t> host_query_items(count * HashSet::kItemWords);
  fill_items<HashSet::kItemWords>(host_insert_items, count);
  fill_items<HashSet::kItemWords>(host_query_items, count,
                                  successful ? 0u : count * 3u + 1u);

  DeviceAllocation<std::uint32_t> insert_items;
  DeviceAllocation<std::uint32_t> query_items;
  DeviceAllocation<std::uint32_t> results;
  DeviceAllocation<std::uint32_t> statuses;
  CudaEvents events;
  if (!insert_items.allocate(count * HashSet::kItemWords, state,
                             "cudaMalloc failed for insert items") ||
      !query_items.allocate(count * HashSet::kItemWords, state,
                            "cudaMalloc failed for query items") ||
      !results.allocate(count, state, "cudaMalloc failed for results") ||
      !statuses.allocate(count, state, "cudaMalloc failed for statuses") ||
      !copy_to_device(insert_items.get(), host_insert_items, state,
                      "cudaMemcpy failed for insert items") ||
      !copy_to_device(query_items.get(), host_query_items, state,
                      "cudaMemcpy failed for query items") ||
      !events.create(state)) {
    return;
  }

  const std::uint32_t buckets = bucket_count_for(count);
  try {
    typename HashSet::Table table(
        buckets,
        overflow_slab_capacity_for(count, buckets, HashSet::kSlotsPerSlab));
    if (HashSet::insert(table, insert_items.get(), count, results.get(),
                        statuses.get()) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("failed to build hash set");
      return;
    }

    for (auto _ : state) {
      if (cudaEventRecord(events.start()) != cudaSuccess) {
        state.SkipWithError("cudaEventRecord(start) failed");
        break;
      }
      if (HashSet::contains(table, query_items.get(), count, results.get(),
                            statuses.get()) != cudaSuccess) {
        state.SkipWithError("hash set contains batch failed");
        break;
      }
      if (!record_elapsed(events.start(), events.stop(), state))
        break;
    }
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(count));
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(count) *
      static_cast<int64_t>(sizeof(std::uint32_t) * (HashSet::kItemWords + 1)));
}

void BM_SlabHashSetInsert(benchmark::State &state) {
  BM_HashSetInsert<SlabHashSetBenchmark>(state);
}

void BM_SlabHashSetContainsSuccessful(benchmark::State &state) {
  BM_HashSetContains<SlabHashSetBenchmark>(state, true);
}

void BM_SlabHashSetContainsUnsuccessful(benchmark::State &state) {
  BM_HashSetContains<SlabHashSetBenchmark>(state, false);
}

void BM_BumpSlabHashSetInsert(benchmark::State &state) {
  BM_HashSetInsert<BumpSlabHashSetBenchmark>(state);
}

void BM_BumpSlabHashSetContainsSuccessful(benchmark::State &state) {
  BM_HashSetContains<BumpSlabHashSetBenchmark>(state, true);
}

void BM_BumpSlabHashSetContainsUnsuccessful(benchmark::State &state) {
  BM_HashSetContains<BumpSlabHashSetBenchmark>(state, false);
}

void BM_AccelerationHashSetInsert(benchmark::State &state) {
  BM_HashSetInsert<AccelerationHashSetBenchmark>(state);
}

void BM_AccelerationHashSetContainsSuccessful(benchmark::State &state) {
  BM_HashSetContains<AccelerationHashSetBenchmark>(state, true);
}

void BM_AccelerationHashSetContainsUnsuccessful(benchmark::State &state) {
  BM_HashSetContains<AccelerationHashSetBenchmark>(state, false);
}

BENCHMARK(BM_SlabHashSetInsert)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();
BENCHMARK(BM_SlabHashSetContainsSuccessful)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();
BENCHMARK(BM_SlabHashSetContainsUnsuccessful)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();

BENCHMARK(BM_BumpSlabHashSetInsert)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();
BENCHMARK(BM_BumpSlabHashSetContainsSuccessful)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();
BENCHMARK(BM_BumpSlabHashSetContainsUnsuccessful)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();

BENCHMARK(BM_AccelerationHashSetInsert)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();
BENCHMARK(BM_AccelerationHashSetContainsSuccessful)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();
BENCHMARK(BM_AccelerationHashSetContainsUnsuccessful)
    ->RangeMultiplier(kRangeMultiplier)
    ->Range(kRangeMin, kRangeMax)
    ->UseManualTime();

} // namespace
