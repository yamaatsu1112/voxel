#pragma once

#include <algo/svt/cuda/device_view.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

namespace algo::svt::cuda {

template <std::uint32_t MaxNodeCount = kDefaultMaxNodeCount,
          std::uint32_t MaxLeafCount = kDefaultMaxLeafCount>
// RAII owner for the device buffers that make up an SVO. Construction only
// allocates memory; callers still need reset_svo() before the tree is usable.
class GpuSvo {
    static_assert(MaxNodeCount > 0u);
    static_assert(MaxLeafCount > 0u);

  public:
    static constexpr std::uint32_t max_node_count = MaxNodeCount;
    static constexpr std::uint32_t max_leaf_count = MaxLeafCount;

    GpuSvo() { status_ = allocate(); }

    ~GpuSvo() { release(); }

    GpuSvo(const GpuSvo&) = delete;
    GpuSvo& operator=(const GpuSvo&) = delete;

    GpuSvo(GpuSvo&& other) noexcept { move_from(other); }

    GpuSvo& operator=(GpuSvo&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    [[nodiscard]] cudaError_t status() const { return status_; }

    // Produce the lightweight non-owning handle that kernels consume.
    [[nodiscard]] DeviceGpuSvo view() const {
        return DeviceGpuSvo{nodes_,
                            leaves_,
                            counters_,
                            free_node_indices_,
                            free_leaf_indices_,
                            MaxNodeCount,
                            MaxLeafCount};
    }

  private:
    cudaError_t allocate() {
        cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&nodes_),
                                        sizeof(GpuSvoNode) * MaxNodeCount);
        if (status != cudaSuccess) {
            release();
            return status;
        }
        status = cudaMalloc(reinterpret_cast<void**>(&leaves_),
                            sizeof(GpuSvoLeaf) * MaxLeafCount);
        if (status != cudaSuccess) {
            release();
            return status;
        }
        status = cudaMalloc(reinterpret_cast<void**>(&counters_),
                            sizeof(GpuSvoCounters));
        if (status != cudaSuccess) {
            release();
            return status;
        }
        status = cudaMalloc(reinterpret_cast<void**>(&free_node_indices_),
                            sizeof(std::uint32_t) * MaxNodeCount);
        if (status != cudaSuccess) {
            release();
            return status;
        }
        status = cudaMalloc(reinterpret_cast<void**>(&free_leaf_indices_),
                            sizeof(std::uint32_t) * MaxLeafCount);
        if (status != cudaSuccess) {
            release();
            return status;
        }
        return cudaSuccess;
    }

    void release() {
        if (nodes_ != nullptr)
            cudaFree(nodes_);
        if (leaves_ != nullptr)
            cudaFree(leaves_);
        if (counters_ != nullptr)
            cudaFree(counters_);
        if (free_node_indices_ != nullptr)
            cudaFree(free_node_indices_);
        if (free_leaf_indices_ != nullptr)
            cudaFree(free_leaf_indices_);
        nodes_ = nullptr;
        leaves_ = nullptr;
        counters_ = nullptr;
        free_node_indices_ = nullptr;
        free_leaf_indices_ = nullptr;
    }

    void move_from(GpuSvo& other) noexcept {
        nodes_ = std::exchange(other.nodes_, nullptr);
        leaves_ = std::exchange(other.leaves_, nullptr);
        counters_ = std::exchange(other.counters_, nullptr);
        free_node_indices_ = std::exchange(other.free_node_indices_, nullptr);
        free_leaf_indices_ = std::exchange(other.free_leaf_indices_, nullptr);
        status_ = std::exchange(other.status_, cudaErrorInvalidValue);
    }

    GpuSvoNode* nodes_ = nullptr;
    GpuSvoLeaf* leaves_ = nullptr;
    GpuSvoCounters* counters_ = nullptr;
    std::uint32_t* free_node_indices_ = nullptr;
    std::uint32_t* free_leaf_indices_ = nullptr;
    cudaError_t status_ = cudaSuccess;
};

} // namespace algo::svt::cuda
