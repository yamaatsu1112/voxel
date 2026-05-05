#pragma once

#include <cstdint>

namespace algo::cuda {

#if defined(__CUDACC__)
#define ALGO_CUDA_HD __host__ __device__
#else
#define ALGO_CUDA_HD
#endif

inline constexpr std::uint32_t kEmptyBucketMarker = 0xffffffffu;

struct DeviceTripleKeyStorage {
    std::uint32_t* key0;
    std::uint32_t* key1;
    std::uint32_t* key2;
    std::uint32_t* value;
};

struct DeviceCuckooTable {
    std::uint64_t* buckets;
    std::uint32_t bucket_count;
    std::uint32_t insertion_attempts;
};

ALGO_CUDA_HD constexpr std::uint64_t pack_bucket_entry(std::uint32_t index,
                                                       std::uint32_t hash_id) {
    return (static_cast<std::uint64_t>(hash_id) << 32) | index;
}

ALGO_CUDA_HD constexpr std::uint32_t unpack_bucket_marker(std::uint64_t packed) {
    return static_cast<std::uint32_t>(packed >> 32);
}

ALGO_CUDA_HD constexpr std::uint32_t unpack_hash_id(std::uint64_t packed) {
    return unpack_bucket_marker(packed);
}

ALGO_CUDA_HD constexpr std::uint32_t unpack_index(std::uint64_t packed) {
    return static_cast<std::uint32_t>(packed & 0xffffffffu);
}

ALGO_CUDA_HD constexpr std::uint32_t
make_key_hash(std::uint32_t a, std::uint32_t b, std::uint32_t c,
              std::uint32_t prime0 = 6000011u,
              std::uint32_t prime1 = 6000023u) {
    const std::uint64_t temp =
        (static_cast<std::uint64_t>(a) * prime0 + b) * prime1 + c;
    return static_cast<std::uint32_t>(temp);
}

ALGO_CUDA_HD constexpr std::uint32_t
hash_location(std::uint32_t a, std::uint32_t b, std::uint32_t c,
              std::uint32_t bucket_count, int hash_id) {
    constexpr std::uint32_t salts[5] = {
        0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu, 0x165667b1u};
    return (make_key_hash(a, b, c) ^ salts[hash_id]) % bucket_count;
}

#ifdef __CUDACC__

#include <cuda_runtime.h>

__device__ inline bool triple_key_equals(const DeviceTripleKeyStorage& keys,
                                         std::uint32_t index, std::uint32_t a,
                                         std::uint32_t b, std::uint32_t c) {
    return keys.key0[index] == a && keys.key1[index] == b &&
           keys.key2[index] == c;
}

template <int NumHashes = 4>
__device__ inline std::uint32_t cuckoo_find(
    const DeviceCuckooTable& table, const DeviceTripleKeyStorage& keys,
    std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    for (int hash_id = 0; hash_id < NumHashes; ++hash_id) {
        const std::uint32_t slot =
            hash_location(a, b, c, table.bucket_count, hash_id);
        const std::uint64_t packed = table.buckets[slot];
        if (unpack_bucket_marker(packed) == kEmptyBucketMarker) continue;
        const std::uint32_t index = unpack_index(packed);
        if (triple_key_equals(keys, index, a, b, c)) return keys.value[index];
    }
    return kEmptyBucketMarker;
}

template <int NumHashes = 4>
__device__ inline bool cuckoo_insert(DeviceCuckooTable table,
                                     const DeviceTripleKeyStorage& keys,
                                     std::uint32_t key_index) {
    std::uint32_t current_hash_id = 0;
    std::uint64_t current = pack_bucket_entry(key_index, current_hash_id);
    std::uint32_t current_index = key_index;
    for (std::uint32_t attempt = 0; attempt < table.insertion_attempts;
         ++attempt) {
        const std::uint32_t a = keys.key0[current_index];
        const std::uint32_t b = keys.key1[current_index];
        const std::uint32_t c = keys.key2[current_index];
        const std::uint32_t slot =
            hash_location(a, b, c, table.bucket_count,
                          static_cast<int>(current_hash_id));
        current = pack_bucket_entry(current_index, current_hash_id);
        const std::uint64_t previous =
            atomicExch(reinterpret_cast<unsigned long long*>(&table.buckets[slot]),
                       static_cast<unsigned long long>(current));
        if (unpack_bucket_marker(previous) == kEmptyBucketMarker) return true;
        current = previous;
        current_index = unpack_index(current);
        current_hash_id = (unpack_hash_id(current) + 1) % NumHashes;
    }
    return false;
}

__global__ void initialize_cuckoo_table(DeviceCuckooTable table) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= table.bucket_count) return;
    table.buckets[index] =
        (static_cast<std::uint64_t>(kEmptyBucketMarker) << 32);
}

#endif

} // namespace algo::cuda

#undef ALGO_CUDA_HD
