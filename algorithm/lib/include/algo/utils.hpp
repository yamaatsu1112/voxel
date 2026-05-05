#pragma once

#include <cstdint>

namespace algo {

constexpr std::uint32_t ceil_div(std::uint32_t num, std::uint32_t den) {
    return den == 0 ? 0 : (num + den - 1) / den;
}

constexpr std::uint32_t next_power_of_two(std::uint32_t value) {
    if (value <= 1) return 1;
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

constexpr bool is_power_of_two(std::uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

} // namespace algo
