// Prefix Sum Utilities
// Provides both global-memory and shared-memory prefix sum implementations.

module prefix_sum;

import utils;

public void reductionShared<let N : int>(
    inout uint sharedData[N],
    uint localIndex,
    uint count
) {
    for (uint offset = count >> 1u; offset > 0u; offset >>= 1u) {
        GroupMemoryBarrierWithGroupSync();
        if (localIndex < offset) {
            sharedData[localIndex] += sharedData[localIndex + offset];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

public void inclusivePrefixSumShared<let N : int, let LANES : int>(
    inout uint sharedData[N],
    uint localIndex,
    uint count
) {
    for (uint offset = 1u; offset < count; offset <<= 1u) {
        uint index = localIndex * (offset << 1u);
        if (index + offset < count) {
            for (uint lane = 0u; lane < LANES; lane++) {
                uint base = lane * count;
                sharedData[base + index] += sharedData[base + index + offset];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint offset = count >> 1u; offset > 0u; offset >>= 1u) {
        uint index = localIndex * (offset << 1u);
        if (index + offset < count) {
            for (uint lane = 0u; lane < LANES; lane++) {
                uint base = lane * count;
                uint temp = sharedData[base + index];
                sharedData[base + index] = temp - sharedData[base + index + offset];
                sharedData[base + index + offset] = temp;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// Shared-memory exclusive prefix sum (Blelloch scan).
// LANES independent prefix sums are executed simultaneously in the same barrier steps.
// sharedData layout: [lane0: count elements][lane1: count elements]...
// count must be a power of two and <= N.
public void exclusivePrefixSumShared<let N : int, let LANES : int>(
    inout uint sharedData[N],
    uint localIndex,
    uint count
) {
    // Up-sweep
    for (uint offset = 1u; offset < count; offset <<= 1u) {
        uint index = (localIndex + 1u) * (offset << 1u) - 1u;
        if (index < count) {
            for (uint lane = 0u; lane < LANES; lane++) {
                uint base = lane * count;
                sharedData[base + index] += sharedData[base + index - offset];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Clear root
    if (localIndex < LANES) {
        sharedData[localIndex * count + count - 1u] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    // Down-sweep
    for (uint offset = count >> 1u; offset > 0u; offset >>= 1u) {
        uint index = (localIndex + 1u) * (offset << 1u) - 1u;
        if (index < count) {
            for (uint lane = 0u; lane < LANES; lane++) {
                uint base = lane * count;
                uint temp = sharedData[base + index - offset];
                sharedData[base + index - offset] = sharedData[base + index];
                sharedData[base + index] += temp;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
