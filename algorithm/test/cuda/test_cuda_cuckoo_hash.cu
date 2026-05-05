#include "cuda_test_utils.cuh"

#include <algo/cuda/cuckoo_hash.cuh>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

using algo::cuda::DeviceCuckooTable;
using algo::cuda::DeviceTripleKeyStorage;

using cuda_test::has_cuda_device;
using cuda_test::managed_alloc;
using cuda_test::sync_cuda;

void init_cuckoo_table(DeviceCuckooTable table) {
    algo::cuda::initialize_cuckoo_table<<<1, 64>>>(table);
    sync_cuda();
}

__global__ void insert_kernel(DeviceCuckooTable table, DeviceTripleKeyStorage keys,
                              std::uint32_t count, bool* ok) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    if (!algo::cuda::cuckoo_insert(table, keys, idx)) *ok = false;
}

__global__ void find_kernel(DeviceCuckooTable table, DeviceTripleKeyStorage keys,
                            const std::uint32_t* q0, const std::uint32_t* q1,
                            const std::uint32_t* q2, std::uint32_t* out,
                            std::uint32_t count) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    out[idx] =
        algo::cuda::cuckoo_find(table, keys, q0[idx], q1[idx], q2[idx]);
}

TEST(CudaCuckooHash, InsertAndFindThreePartKeys) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    constexpr std::uint32_t kCount = 3;
    constexpr std::uint32_t kBuckets = 32;

    auto* buckets = managed_alloc<std::uint64_t>(kBuckets);
    auto* key0 = managed_alloc<std::uint32_t>(kCount);
    auto* key1 = managed_alloc<std::uint32_t>(kCount);
    auto* key2 = managed_alloc<std::uint32_t>(kCount);
    auto* values = managed_alloc<std::uint32_t>(kCount);
    auto* results = managed_alloc<std::uint32_t>(kCount);
    auto* ok = managed_alloc<bool>(1);
    ASSERT_NE(buckets, nullptr);
    ASSERT_NE(key0, nullptr);
    ASSERT_NE(key1, nullptr);
    ASSERT_NE(key2, nullptr);
    ASSERT_NE(values, nullptr);
    ASSERT_NE(results, nullptr);
    ASSERT_NE(ok, nullptr);

    DeviceTripleKeyStorage keys{key0, key1, key2, values};
    DeviceCuckooTable table{buckets, kBuckets, 16};

    key0[0] = 7;
    key1[0] = 11;
    key2[0] = 13;
    values[0] = 100;

    key0[1] = 17;
    key1[1] = 19;
    key2[1] = 23;
    values[1] = 200;

    key0[2] = 29;
    key1[2] = 31;
    key2[2] = 37;
    values[2] = 300;
    *ok = true;

    init_cuckoo_table(table);

    insert_kernel<<<1, 64>>>(table, keys, kCount, ok);
    sync_cuda();
    ASSERT_TRUE(*ok);

    find_kernel<<<1, 64>>>(table, keys, key0, key1, key2, results, kCount);
    sync_cuda();

    EXPECT_EQ(results[0], 100u);
    EXPECT_EQ(results[1], 200u);
    EXPECT_EQ(results[2], 300u);

    cudaFree(ok);
    cudaFree(results);
    cudaFree(values);
    cudaFree(key2);
    cudaFree(key1);
    cudaFree(key0);
    cudaFree(buckets);
}

TEST(CudaCuckooHash, InsertAndFindAfterEviction) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    constexpr std::uint32_t kCount = 2;
    // Use a non-power-of-two bucket count so different salts can produce
    // distinct secondary locations for keys that collide in the first slot.
    constexpr std::uint32_t kBuckets = 31;

    std::array<std::array<std::uint32_t, 3>, kCount> host_keys{};
    bool found = false;
    for (std::uint32_t a0 = 1; a0 < 128 && !found; ++a0) {
        for (std::uint32_t b0 = 1; b0 < 128 && !found; ++b0) {
            for (std::uint32_t c0 = 1; c0 < 128 && !found; ++c0) {
                const auto h0 = algo::cuda::hash_location(a0, b0, c0, kBuckets, 0);
                const auto h1 = algo::cuda::hash_location(a0, b0, c0, kBuckets, 1);
                for (std::uint32_t a1 = a0; a1 < 128 && !found; ++a1) {
                    for (std::uint32_t b1 = 1; b1 < 128 && !found; ++b1) {
                        for (std::uint32_t c1 = 1; c1 < 128 && !found; ++c1) {
                            if (a0 == a1 && b0 == b1 && c0 == c1) continue;
                            if (algo::cuda::hash_location(a1, b1, c1, kBuckets, 0) !=
                                h0) {
                                continue;
                            }
                            if (algo::cuda::hash_location(a1, b1, c1, kBuckets, 1) ==
                                h1) {
                                continue;
                            }
                            host_keys[0] = {a0, b0, c0};
                            host_keys[1] = {a1, b1, c1};
                            found = true;
                        }
                    }
                }
            }
        }
    }
    ASSERT_TRUE(found);

    auto* buckets = managed_alloc<std::uint64_t>(kBuckets);
    auto* key0 = managed_alloc<std::uint32_t>(kCount);
    auto* key1 = managed_alloc<std::uint32_t>(kCount);
    auto* key2 = managed_alloc<std::uint32_t>(kCount);
    auto* values = managed_alloc<std::uint32_t>(kCount);
    auto* results = managed_alloc<std::uint32_t>(kCount);
    auto* ok = managed_alloc<bool>(1);
    ASSERT_NE(buckets, nullptr);
    ASSERT_NE(key0, nullptr);
    ASSERT_NE(key1, nullptr);
    ASSERT_NE(key2, nullptr);
    ASSERT_NE(values, nullptr);
    ASSERT_NE(results, nullptr);
    ASSERT_NE(ok, nullptr);

    DeviceTripleKeyStorage keys{key0, key1, key2, values};
    DeviceCuckooTable table{buckets, kBuckets, 16};
    for (std::uint32_t i = 0; i < kCount; ++i) {
        key0[i] = host_keys[i][0];
        key1[i] = host_keys[i][1];
        key2[i] = host_keys[i][2];
        values[i] = 100u + i;
    }
    *ok = true;

    init_cuckoo_table(table);

    insert_kernel<<<1, 64>>>(table, keys, kCount, ok);
    sync_cuda();
    ASSERT_TRUE(*ok);

    find_kernel<<<1, 64>>>(table, keys, key0, key1, key2, results, kCount);
    sync_cuda();

    EXPECT_EQ(results[0], 100u);
    EXPECT_EQ(results[1], 101u);

    cudaFree(ok);
    cudaFree(results);
    cudaFree(values);
    cudaFree(key2);
    cudaFree(key1);
    cudaFree(key0);
    cudaFree(buckets);
}

} // namespace
