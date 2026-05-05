#pragma once

#include <cstddef>

namespace algo::svt::cuda::detail {

template <class T> constexpr std::size_t align_up(std::size_t offset) {
    constexpr std::size_t kAlign = alignof(T);
    return (offset + kAlign - 1) & ~(kAlign - 1);
}

template <class T> T* pointer_at(void* base, std::size_t offset) {
    return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <class T>
inline void reserve_array(void* workspace, std::size_t count,
                          std::size_t& offset, T*& out) {
    offset = align_up<T>(offset);
    out = pointer_at<T>(workspace, offset);
    offset += sizeof(T) * count;
}

template <class T>
inline void reserve_workspace(void* workspace, std::size_t size,
                              std::size_t& offset, void*& out) {
    offset = align_up<T>(offset);
    out = pointer_at<std::byte>(workspace, offset);
    offset += size;
}

} // namespace algo::svt::cuda::detail
