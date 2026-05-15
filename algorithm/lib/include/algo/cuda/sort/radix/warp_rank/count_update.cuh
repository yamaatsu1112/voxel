#pragma once

#include <cstddef>
#include <cstdint>

namespace algo::cuda::sort::detail {

struct warp_rank_count_store {
    __device__ __forceinline__ static void
    update(std::uint32_t* counts, std::size_t index, std::uint32_t count) {
        counts[index] = count;
    }
};

struct warp_rank_count_add {
    __device__ __forceinline__ static void
    update(std::uint32_t* counts, std::size_t index, std::uint32_t count) {
        counts[index] += count;
    }
};

} // namespace algo::cuda::sort::detail
