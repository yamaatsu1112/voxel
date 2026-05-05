// Utils - General Purpose Utility Functions
// This module contains utility functions that are not specific to SVO operations.

module utils;

// ============================================================================
// Math Utility Functions
// ============================================================================

// Calculate next power of 2 for array size
public uint nextPowerOf2(uint n) {
    if (n == 0u) return 1u;
    n--;
    n |= n >> 1u;
    n |= n >> 2u;
    n |= n >> 4u;
    n |= n >> 8u;
    n |= n >> 16u;
    n++;
    return n;
}

// Calculate floor(log2(n)) for n > 0
public uint log2Floor(uint n) {
    if (n == 0u) return 0u;
    uint result = 0u;
    if (n >= (1u << 16u)) { result += 16u; n >>= 16u; }
    if (n >= (1u << 8u)) { result += 8u; n >>= 8u; }
    if (n >= (1u << 4u)) { result += 4u; n >>= 4u; }
    if (n >= (1u << 2u)) { result += 2u; n >>= 2u; }
    if (n >= (1u << 1u)) { result += 1u; }
    return result;
}
