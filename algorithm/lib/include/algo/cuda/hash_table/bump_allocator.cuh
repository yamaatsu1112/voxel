#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#endif

namespace algo::cuda {

#ifdef __CUDACC__

struct BumpSlabAllocator {
  struct Device {
    std::uint32_t *next = nullptr;
    std::uint32_t slab_capacity = 0;
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

    void allocate(std::uint32_t slab_capacity_value) {
      release();
      const cudaError_t status =
          cudaMalloc(reinterpret_cast<void **>(&next), sizeof(std::uint32_t));
      if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                 cudaGetErrorString(status));
      }
      slab_capacity = slab_capacity_value;
    }

    void release() noexcept {
      if (next != nullptr)
        cudaFree(next);
      next = nullptr;
      slab_capacity = 0;
    }

    [[nodiscard]] Device device_view() const {
      return Device{next, slab_capacity};
    }

    std::uint32_t *next = nullptr;
    std::uint32_t slab_capacity = 0;

  private:
    void move_from(Host &other) noexcept {
      next = std::exchange(other.next, nullptr);
      slab_capacity = std::exchange(other.slab_capacity, 0);
    }
  };

  __device__ static void initialize(Device allocator) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
      *allocator.next = 0;
  }

  __device__ static void initialize(Device allocator, std::uint32_t first) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
      *allocator.next = first;
  }

  __device__ static std::uint32_t allocate(Device allocator,
                                           std::uint32_t lane) {
    constexpr std::uint32_t kFullWarpMask = 0xffffffffu;
    constexpr std::uint32_t kNullSlab = 0xffffffffu;
    std::uint32_t dynamic_ordinal = kNullSlab;
    if (lane == 0)
      dynamic_ordinal = atomicAdd(allocator.next, 1u);
    dynamic_ordinal = __shfl_sync(kFullWarpMask, dynamic_ordinal, 0);
    if (dynamic_ordinal >= allocator.slab_capacity)
      return kNullSlab;

    return dynamic_ordinal;
  }

  template <class Slab>
  __device__ static std::uint32_t allocate(Device allocator, Slab *slabs,
                                           std::uint32_t slab_count,
                                           std::uint32_t lane) {
    constexpr std::uint32_t kFullWarpMask = 0xffffffffu;
    constexpr std::uint32_t kNullSlab = 0xffffffffu;
    std::uint32_t index = kNullSlab;
    if (lane == 0)
      index = atomicAdd(allocator.next, 1u);
    index = __shfl_sync(kFullWarpMask, index, 0);
    if (index >= slab_count)
      return kNullSlab;

    Slab::initialize(&slabs[index], lane);
    __syncwarp(kFullWarpMask);
    __threadfence();
    __syncwarp(kFullWarpMask);
    return index;
  }

  __device__ static void deallocate(Device, std::uint32_t, std::uint32_t) {
    // Bump allocation is monotonic. Slabs allocated before losing a publish
    // CAS intentionally leak; lost or freed slabs are not recycled.
  }
};

#endif

} // namespace algo::cuda
