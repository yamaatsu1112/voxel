#pragma once

#include <algo/cuda/hash_table/acceleration_hash/set/kernels.cuh>

#ifdef __CUDACC__

#include <algo/cuda/utils.cuh>

#include <limits>
#include <stdexcept>
#include <utility>

namespace algo::cuda {

template <std::uint32_t ItemWords, class Allocator = BumpSlabAllocator>
class AccelerationHashSet32 {
public:
  using DeviceView = DeviceAccelerationHashSet32<ItemWords, Allocator>;
  using Slab = AccelerationHashSlab32<ItemWords>;

  AccelerationHashSet32(std::uint32_t bucket_count,
                        std::uint32_t overflow_slab_capacity)
      : bucket_count_(bucket_count),
        overflow_slab_capacity_(overflow_slab_capacity) {
    if (bucket_count == 0)
      throw std::invalid_argument("AccelerationHashSet32 requires buckets");
    if (overflow_slab_capacity >
        std::numeric_limits<std::uint32_t>::max() - bucket_count) {
      throw std::invalid_argument("AccelerationHashSet32 slab count overflow");
    }
    slab_count_ = bucket_count_ + overflow_slab_capacity_;
    slabs_ = device_alloc<Slab>(slab_count_);
    allocator_.allocate(slab_count_);
    clear();
  }

  AccelerationHashSet32(const AccelerationHashSet32 &) = delete;
  AccelerationHashSet32 &operator=(const AccelerationHashSet32 &) = delete;

  AccelerationHashSet32(AccelerationHashSet32 &&other) noexcept {
    move_from(other);
  }

  AccelerationHashSet32 &operator=(AccelerationHashSet32 &&other) noexcept {
    if (this != &other) {
      release();
      move_from(other);
    }
    return *this;
  }

  ~AccelerationHashSet32() { release(); }

  [[nodiscard]] DeviceView device_view() const {
    return DeviceView{slabs_, bucket_count_, slab_count_,
                      allocator_.device_view()};
  }

  [[nodiscard]] std::uint32_t bucket_count() const { return bucket_count_; }
  [[nodiscard]] std::uint32_t slab_capacity() const {
    return overflow_slab_capacity_;
  }
  [[nodiscard]] std::uint32_t slab_count() const { return slab_count_; }

  void clear(cudaStream_t stream = nullptr) {
    check_cuda(initialize_acceleration_hash_set32(device_view(), stream),
               "initialize_acceleration_hash_set32 launch failed");
    check_cuda(cudaStreamSynchronize(stream),
               "initialize_acceleration_hash_set32 synchronize failed");
  }

  // Looks up currently published table contents. This is not an
  // insert-if-absent primitive and does not guarantee uniqueness against
  // concurrent unchecked inserts.
  cudaError_t find_batch(const std::uint32_t *items, std::uint32_t count,
                         std::uint32_t *ptrs, std::uint32_t *statuses = nullptr,
                         cudaStream_t stream = nullptr) const {
    return acceleration_hash_set32_find_batch(device_view(), items, count, ptrs,
                                              statuses, stream);
  }

  // Inserts without checking for an equal item. The caller must ensure that no
  // equal item is already present or concurrently inserted.
  cudaError_t insert_unique_unchecked_batch(const std::uint32_t *items,
                                           std::uint32_t count,
                                           std::uint32_t *ptrs,
                                           std::uint32_t *statuses = nullptr,
                                           cudaStream_t stream = nullptr) {
    return acceleration_hash_set32_insert_unique_unchecked_batch(
        device_view(), items, count, ptrs, statuses, stream);
  }

private:
  void release() noexcept {
    allocator_.release();
    if (slabs_ != nullptr)
      cudaFree(slabs_);
    slabs_ = nullptr;
    bucket_count_ = 0;
    overflow_slab_capacity_ = 0;
    slab_count_ = 0;
  }

  void move_from(AccelerationHashSet32 &other) noexcept {
    slabs_ = std::exchange(other.slabs_, nullptr);
    allocator_ = std::move(other.allocator_);
    bucket_count_ = std::exchange(other.bucket_count_, 0);
    overflow_slab_capacity_ = std::exchange(other.overflow_slab_capacity_, 0);
    slab_count_ = std::exchange(other.slab_count_, 0);
  }

  Slab *slabs_ = nullptr;
  std::uint32_t bucket_count_ = 0;
  std::uint32_t overflow_slab_capacity_ = 0;
  std::uint32_t slab_count_ = 0;
  typename Allocator::Host allocator_;
};

} // namespace algo::cuda

#endif
