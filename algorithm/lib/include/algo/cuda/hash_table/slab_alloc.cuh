#pragma once

#ifdef __CUDACC__

#include <algo/cuda/hash_table/slab_hash/common.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#endif

namespace algo::cuda {

#ifdef __CUDACC__

inline constexpr std::uint32_t kSlabAllocBlockSlabs = 1024;
inline constexpr std::uint32_t kSlabAllocBitmapBitsPerWord = 32;
inline constexpr std::uint32_t kSlabAllocBitmapWordsPerBlock =
    kSlabAllocBlockSlabs / kSlabAllocBitmapBitsPerWord;
inline constexpr std::uint32_t kSlabAllocDefaultResidentWarpCount = 1u << 18;

struct SlabAlloc {
  struct Device {
    std::uint32_t *bitmap_words = nullptr;
    std::uint32_t *resident_change_attempts = nullptr;
    std::uint32_t slab_capacity = 0;
    std::uint32_t block_count = 0;
  };

  struct Host {
    Host() = default;

    Host(const Host &) = delete;
    Host &operator=(const Host &) = delete;

    Host(Host &&other) noexcept { move_from(other); }

    Host &operator=(Host &&other) noexcept {
      if (this != &other) {
        release();
        move_from(other);
      }
      return *this;
    }

    ~Host() { release(); }

    void allocate(std::uint32_t slab_capacity_value) {
      release();

      slab_capacity = slab_capacity_value;
      block_count = block_count_for(slab_capacity_value);
      const std::uint32_t words_to_allocate =
          bitmap_word_count_for(slab_capacity_value);

      if (words_to_allocate != 0) {
        const cudaError_t bitmap_status =
            cudaMalloc(reinterpret_cast<void **>(&bitmap_words),
                       sizeof(std::uint32_t) * words_to_allocate);
        if (bitmap_status != cudaSuccess) {
          reset_fields();
          throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                   cudaGetErrorString(bitmap_status));
        }
      }

      const cudaError_t resident_status = cudaMalloc(
          reinterpret_cast<void **>(&resident_change_attempts),
          sizeof(std::uint32_t) * kSlabAllocDefaultResidentWarpCount);
      if (resident_status != cudaSuccess) {
        release();
        throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                 cudaGetErrorString(resident_status));
      }
    }

    void release() noexcept {
      if (resident_change_attempts != nullptr)
        cudaFree(resident_change_attempts);
      if (bitmap_words != nullptr)
        cudaFree(bitmap_words);
      reset_fields();
    }

    [[nodiscard]] Device device_view() const {
      return Device{bitmap_words, resident_change_attempts, slab_capacity,
                    block_count};
    }

    __host__ __device__ static constexpr std::uint32_t
    bitmap_word_count_for(std::uint32_t slab_capacity_value) {
      return (slab_capacity_value + kSlabAllocBitmapBitsPerWord - 1u) /
             kSlabAllocBitmapBitsPerWord;
    }

    __host__ __device__ static constexpr std::uint32_t
    block_count_for(std::uint32_t slab_capacity_value) {
      return (slab_capacity_value + kSlabAllocBlockSlabs - 1u) /
             kSlabAllocBlockSlabs;
    }

    std::uint32_t *bitmap_words = nullptr;
    std::uint32_t *resident_change_attempts = nullptr;
    std::uint32_t slab_capacity = 0;
    std::uint32_t block_count = 0;

  private:
    void move_from(Host &other) noexcept {
      bitmap_words = std::exchange(other.bitmap_words, nullptr);
      resident_change_attempts =
          std::exchange(other.resident_change_attempts, nullptr);
      slab_capacity = std::exchange(other.slab_capacity, 0);
      block_count = std::exchange(other.block_count, 0);
    }

    void reset_fields() noexcept {
      bitmap_words = nullptr;
      resident_change_attempts = nullptr;
      slab_capacity = 0;
      block_count = 0;
    }
  };

  __device__ static void initialize(Device allocator) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = blockDim.x * gridDim.x;
    const std::uint32_t bitmap_word_count =
        Host::bitmap_word_count_for(allocator.slab_capacity);

    for (std::uint32_t word = index; word < bitmap_word_count; word += stride)
      allocator.bitmap_words[word] = 0;

    for (std::uint32_t resident = index;
         resident < kSlabAllocDefaultResidentWarpCount; resident += stride) {
      allocator.resident_change_attempts[resident] = 0;
    }
  }

  __device__ static std::uint32_t allocate(Device allocator,
                                           std::uint32_t lane) {
    constexpr std::uint32_t kFullWarpMask = 0xffffffffu;
    if (allocator.slab_capacity == 0 || allocator.block_count == 0 ||
        allocator.bitmap_words == nullptr ||
        allocator.resident_change_attempts == nullptr)
      return kSlabHashNullSlab;

    const std::uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t global_warp = global_thread / 32u;
    const std::uint32_t resident_slot =
        global_warp % kSlabAllocDefaultResidentWarpCount;
    std::uint32_t attempt = allocator.resident_change_attempts[resident_slot];
    const std::uint32_t first_block =
        resident_block_for(global_warp, attempt, allocator.block_count);

    for (std::uint32_t probe = 0; probe < allocator.block_count; ++probe) {
      const std::uint32_t block = (first_block + probe) % allocator.block_count;
      const std::uint32_t allocated =
          allocate_from_block(allocator, block, lane);
      if (allocated != kSlabHashNullSlab)
        return allocated;

      std::uint32_t next_attempt = attempt;
      if (lane == 0) {
        next_attempt =
            atomicAdd(&allocator.resident_change_attempts[resident_slot], 1u) +
            1u;
      }
      attempt = __shfl_sync(kFullWarpMask, next_attempt, 0);
    }

    return kSlabHashNullSlab;
  }

  __device__ static void deallocate(Device allocator,
                                    std::uint32_t dynamic_slab_ordinal,
                                    std::uint32_t lane) {
    if (lane != 0 || dynamic_slab_ordinal >= allocator.slab_capacity ||
        allocator.bitmap_words == nullptr)
      return;

    const std::uint32_t word =
        dynamic_slab_ordinal / kSlabAllocBitmapBitsPerWord;
    const std::uint32_t bit =
        dynamic_slab_ordinal % kSlabAllocBitmapBitsPerWord;
    atomicAnd(&allocator.bitmap_words[word], ~(1u << bit));
  }

private:
  __device__ static std::uint32_t
  resident_block_for(std::uint32_t global_warp, std::uint32_t attempt,
                     std::uint32_t block_count) {
    return slab_hash_mix(global_warp ^ (attempt * 0x9e3779b9u)) % block_count;
  }

  __device__ static std::uint32_t allocate_from_block(Device allocator,
                                                      std::uint32_t block,
                                                      std::uint32_t lane) {
    constexpr std::uint32_t kFullWarpMask = 0xffffffffu;
    const std::uint32_t word_index =
        block * kSlabAllocBitmapWordsPerBlock + lane;
    const std::uint32_t block_base = block * kSlabAllocBlockSlabs;
    const std::uint32_t word_base =
        block_base + lane * kSlabAllocBitmapBitsPerWord;

    for (std::uint32_t retry = 0; retry < kSlabAllocBlockSlabs; ++retry) {
      std::uint32_t free_bits = 0;
      if (word_base < allocator.slab_capacity) {
        const std::uint32_t remaining = allocator.slab_capacity - word_base;
        const std::uint32_t valid_bits =
            remaining >= kSlabAllocBitmapBitsPerWord ? 0xffffffffu
                                                     : ((1u << remaining) - 1u);
        free_bits = ~allocator.bitmap_words[word_index] & valid_bits;
      }

      const std::uint32_t lanes_with_free =
          __ballot_sync(kFullWarpMask, free_bits != 0);
      if (lanes_with_free == 0)
        return kSlabHashNullSlab;

      const std::uint32_t selected_lane =
          static_cast<std::uint32_t>(__ffs(lanes_with_free) - 1);
      const std::uint32_t selected_free_bits =
          __shfl_sync(kFullWarpMask, free_bits, selected_lane);
      const std::uint32_t selected_bit =
          static_cast<std::uint32_t>(__ffs(selected_free_bits) - 1);
      const std::uint32_t selected_word =
          block * kSlabAllocBitmapWordsPerBlock + selected_lane;
      const std::uint32_t bit_mask = 1u << selected_bit;

      std::uint32_t acquired = 0;
      if (lane == selected_lane) {
        const std::uint32_t old =
            atomicOr(&allocator.bitmap_words[selected_word], bit_mask);
        acquired = (old & bit_mask) == 0 ? 1u : 0u;
      }
      acquired = __shfl_sync(kFullWarpMask, acquired, selected_lane);
      if (acquired == 0)
        continue;

      const std::uint32_t ordinal =
          block * kSlabAllocBlockSlabs +
          selected_lane * kSlabAllocBitmapBitsPerWord + selected_bit;
      return ordinal < allocator.slab_capacity ? ordinal : kSlabHashNullSlab;
    }
    return kSlabHashNullSlab;
  }
};

#endif

} // namespace algo::cuda
