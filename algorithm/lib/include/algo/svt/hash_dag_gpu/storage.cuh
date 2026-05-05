#pragma once

#include <algo/cuda/acceleration_hash.cuh>
#include <algo/cuda/utils.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace algo::svt::hash_dag_gpu {

// Host owner for the GPU hash DAG. The DAG itself is append-only/canonicalized
// through hash tables; reset() clears all tables and restores the root to empty.
template <std::uint32_t BucketCount = kDefaultBucketCount,
          std::uint32_t OverflowSlabCount = kDefaultOverflowSlabCount>
class HashDagGpu {
public:
  using Allocator = DeviceHashDagGpu::Allocator;

  HashDagGpu()
      : table1_(BucketCount, OverflowSlabCount),
        table2_(BucketCount, OverflowSlabCount),
        table3_(BucketCount, OverflowSlabCount),
        table4_(BucketCount, OverflowSlabCount),
        table5_(BucketCount, OverflowSlabCount),
        table6_(BucketCount, OverflowSlabCount),
        table7_(BucketCount, OverflowSlabCount),
        table8_(BucketCount, OverflowSlabCount),
        table9_(BucketCount, OverflowSlabCount) {
    status_ = cudaMalloc(reinterpret_cast<void **>(&root_ref_),
                         sizeof(std::uint32_t));
    if (status_ == cudaSuccess)
      status_ = reset_root();
  }

  ~HashDagGpu() { release(); }

  HashDagGpu(const HashDagGpu &) = delete;
  HashDagGpu &operator=(const HashDagGpu &) = delete;
  HashDagGpu(HashDagGpu &&) = delete;
  HashDagGpu &operator=(HashDagGpu &&) = delete;

  [[nodiscard]] cudaError_t status() const { return status_; }

  // Build the lightweight value passed to kernels and device functions. The
  // returned view does not own storage and is only valid while this object lives.
  [[nodiscard]] DeviceHashDagGpu view() const {
    return DeviceHashDagGpu{root_ref_,
                            table1_.device_view(),
                            table2_.device_view(),
                            table3_.device_view(),
                            table4_.device_view(),
                            table5_.device_view(),
                            table6_.device_view(),
                            table7_.device_view(),
                            table8_.device_view(),
                            table9_.device_view()};
  }

  cudaError_t reset(cudaStream_t stream = nullptr) {
    table1_.clear(stream);
    table2_.clear(stream);
    table3_.clear(stream);
    table4_.clear(stream);
    table5_.clear(stream);
    table6_.clear(stream);
    table7_.clear(stream);
    table8_.clear(stream);
    table9_.clear(stream);
    return reset_root(stream);
  }

  [[nodiscard]] std::uint32_t bucket_count() const { return BucketCount; }

  [[nodiscard]] std::uint32_t overflow_slab_count() const {
    return OverflowSlabCount;
  }

private:
  cudaError_t reset_root(cudaStream_t stream = nullptr) {
    const std::uint32_t empty = kEmptyRef;
    return cudaMemcpyAsync(root_ref_, &empty, sizeof(empty),
                           cudaMemcpyHostToDevice, stream);
  }

  void release() noexcept {
    if (root_ref_ != nullptr)
      cudaFree(root_ref_);
    root_ref_ = nullptr;
  }

  std::uint32_t *root_ref_ = nullptr;
  algo::cuda::AccelerationHashSet32<1, Allocator> table1_;
  algo::cuda::AccelerationHashSet32<2, Allocator> table2_;
  algo::cuda::AccelerationHashSet32<3, Allocator> table3_;
  algo::cuda::AccelerationHashSet32<4, Allocator> table4_;
  algo::cuda::AccelerationHashSet32<5, Allocator> table5_;
  algo::cuda::AccelerationHashSet32<6, Allocator> table6_;
  algo::cuda::AccelerationHashSet32<7, Allocator> table7_;
  algo::cuda::AccelerationHashSet32<8, Allocator> table8_;
  algo::cuda::AccelerationHashSet32<9, Allocator> table9_;
  cudaError_t status_ = cudaSuccess;
};

} // namespace algo::svt::hash_dag_gpu
