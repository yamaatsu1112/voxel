#pragma once

#include <algo/cuda/hash_table/slab_alloc.cuh>
#include <algo/cuda/hash_table/slab_hash/set/kernels.cuh>

#ifdef __CUDACC__

#include <algo/cuda/utils.cuh>

#include <stdexcept>
#include <utility>

namespace algo::cuda {

template <class Allocator> class BasicSlabHashSetU32 {
public:
  using DeviceView = DeviceSlabHashSetU32<Allocator>;

  BasicSlabHashSetU32(std::uint32_t bucket_count, std::uint32_t slab_capacity)
      : bucket_count_(bucket_count), slab_capacity_(slab_capacity) {
    if (bucket_count == 0) {
      throw std::invalid_argument("BasicSlabHashSetU32 requires buckets");
    }
    slabs_ = device_alloc<SlabHashSetU32Slab>(
        slab_hash_slab_storage_count(bucket_count_, slab_capacity_));
    allocator_.allocate(slab_capacity_);
    clear();
  }

  BasicSlabHashSetU32(const BasicSlabHashSetU32 &) = delete;
  BasicSlabHashSetU32 &operator=(const BasicSlabHashSetU32 &) = delete;

  BasicSlabHashSetU32(BasicSlabHashSetU32 &&other) noexcept {
    move_from(other);
  }

  BasicSlabHashSetU32 &operator=(BasicSlabHashSetU32 &&other) noexcept {
    if (this != &other) {
      release();
      move_from(other);
    }
    return *this;
  }

  ~BasicSlabHashSetU32() { release(); }

  [[nodiscard]] DeviceView device_view() const {
    return DeviceView{slabs_, bucket_count_, slab_capacity_,
                      allocator_.device_view()};
  }

  [[nodiscard]] std::uint32_t bucket_count() const { return bucket_count_; }
  [[nodiscard]] std::uint32_t slab_capacity() const { return slab_capacity_; }

  void clear(cudaStream_t stream = nullptr) {
    check_cuda(initialize_slab_hash_set(device_view(), stream),
               "initialize_slab_hash_set launch failed");
    check_cuda(cudaStreamSynchronize(stream),
               "initialize_slab_hash_set synchronize failed");
  }

  cudaError_t insert_batch(const std::uint32_t *keys, std::uint32_t count,
                           std::uint32_t *statuses = nullptr,
                           cudaStream_t stream = nullptr) {
    return slab_hash_set_insert_batch(device_view(), keys, count, statuses,
                                      stream);
  }

  cudaError_t contains_batch(const std::uint32_t *keys, std::uint32_t *contains,
                             std::uint32_t count,
                             std::uint32_t *statuses = nullptr,
                             cudaStream_t stream = nullptr) const {
    return slab_hash_set_contains_batch(device_view(), keys, contains, count,
                                        statuses, stream);
  }

  cudaError_t erase_batch(const std::uint32_t *keys, std::uint32_t count,
                          std::uint32_t *statuses = nullptr,
                          cudaStream_t stream = nullptr) {
    return slab_hash_set_erase_batch(device_view(), keys, count, statuses,
                                     stream);
  }

private:
  void release() noexcept {
    allocator_.release();
    if (slabs_ != nullptr)
      cudaFree(slabs_);
    slabs_ = nullptr;
    bucket_count_ = 0;
    slab_capacity_ = 0;
  }

  void move_from(BasicSlabHashSetU32 &other) noexcept {
    slabs_ = std::exchange(other.slabs_, nullptr);
    allocator_ = std::move(other.allocator_);
    bucket_count_ = std::exchange(other.bucket_count_, 0);
    slab_capacity_ = std::exchange(other.slab_capacity_, 0);
  }

  SlabHashSetU32Slab *slabs_ = nullptr;
  std::uint32_t bucket_count_ = 0;
  std::uint32_t slab_capacity_ = 0;
  typename Allocator::Host allocator_;
};

using SlabHashSetU32 = BasicSlabHashSetU32<SlabAlloc>;

} // namespace algo::cuda

#endif
