#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef __CUDACC__
#error "algo::cuda::sort::UIntKey requires CUDA compilation with nvcc"
#endif

#include <cuda_runtime.h>

namespace algo::cuda::sort {

template <std::size_t Words> struct UIntKey {
    static_assert(Words > 0, "UIntKey requires Words > 0");

    std::uint32_t words[Words]{};

    __host__ __device__ friend bool operator==(const UIntKey& lhs,
                                               const UIntKey& rhs) {
        for (std::size_t i = 0; i < Words; ++i) {
            if (lhs.words[i] != rhs.words[i])
                return false;
        }
        return true;
    }
};

template <class Key, class = void> struct radix_key_traits {
    static constexpr bool kSupported = false;
};

template <class Key>
struct radix_key_traits<
    Key, std::enable_if_t<std::is_integral_v<Key> && std::is_unsigned_v<Key>>> {
    static constexpr bool kSupported = true;
    static constexpr int kBits = static_cast<int>(sizeof(Key) * 8u);

    template <int RadixBits>
    __host__ __device__ static std::uint32_t digit(Key key, int shift) {
        constexpr Key kMask = (Key{1} << RadixBits) - Key{1};
        return static_cast<std::uint32_t>((key >> shift) & kMask);
    }
};

template <std::size_t Words> struct radix_key_traits<UIntKey<Words>> {
    static constexpr bool kSupported = true;
    static constexpr int kBits = static_cast<int>(Words * 32u);

    template <int RadixBits>
    __host__ __device__ static std::uint32_t digit(UIntKey<Words> key,
                                                   int shift) {
        constexpr std::uint32_t kMask =
            static_cast<std::uint32_t>((1u << RadixBits) - 1u);
        const std::size_t word = static_cast<std::size_t>(shift / 32);
        const int bit = shift % 32;
        std::uint32_t value = word < Words ? key.words[word] >> bit : 0u;
        if constexpr (RadixBits > 1) {
            if (bit + RadixBits > 32 && word + 1 < Words) {
                value |= key.words[word + 1] << (32 - bit);
            }
        }
        return value & kMask;
    }
};

} // namespace algo::cuda::sort
