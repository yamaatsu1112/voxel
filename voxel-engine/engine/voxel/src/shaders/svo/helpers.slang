// SVO Helpers - Helper Functions for SVO Operations
// This module contains helper functions for SVO traversal and coordinate calculation.

implementing svo;

// ============================================================================
// Helper Functions (free functions, SVO-independent)
// ============================================================================

public uint makeMirrorMask1BitPerAxis(int3 negDirection) {
    return
        (negDirection.x << 0) |
        (negDirection.y << 1) |
        (negDirection.z << 2);
}

public uint makeMirrorMask2BitsPerAxis(int3 negDirection) {
    return
        (negDirection.x << 0) | (negDirection.x << 1) |
        (negDirection.y << 2) | (negDirection.y << 3) |
        (negDirection.z << 4) | (negDirection.z << 5);
}

// Calculate child index from voxel coordinates at given depth
public uint getChildIndex(uint x, uint y, uint z, uint depth) {
    uint shift = (MAX_DEPTH - depth - 1u) * BRANCH_FACTOR_EXP + LEAF_VOXEL_COUNT_EXP;
    uint mask = (1u << BRANCH_FACTOR_EXP) - 1u;
    uint xBits = (x >> shift) & mask;
    uint yBits = (y >> shift) & mask;
    uint zBits = (z >> shift) & mask;
    return xBits | (yBits << BRANCH_FACTOR_EXP) | (zBits << (BRANCH_FACTOR_EXP * 2u));
}

// Calculate leaf offset from voxel coordinates
public uint getLeafOffset(uint x, uint y, uint z) {
    uint xOffset = x & ((1u << LEAF_VOXEL_COUNT_EXP) - 1);
    uint yOffset = y & ((1u << LEAF_VOXEL_COUNT_EXP) - 1);
    uint zOffset = z & ((1u << LEAF_VOXEL_COUNT_EXP) - 1);
    return xOffset + (yOffset << LEAF_VOXEL_COUNT_EXP) +
           (zOffset << (LEAF_VOXEL_COUNT_EXP << 1));
}

// Calculate surface normal from face index
public float3 getSurfaceNormal(Face face) {
    switch (face) {
        case Face::X_POS: return float3(1.0, 0.0, 0.0);
        case Face::X_NEG: return float3(-1.0, 0.0, 0.0);
        case Face::Y_POS: return float3(0.0, 1.0, 0.0);
        case Face::Y_NEG: return float3(0.0, -1.0, 0.0);
        case Face::Z_POS: return float3(0.0, 0.0, 1.0);
        case Face::Z_NEG: return float3(0.0, 0.0, -1.0);
        default: return float3(0.0, 0.0, 0.0);
    }
}

public float3 calculateRayDirection(
    UniformBufferObject ubo,
    float2 screenCoord,
    float3 rayOrigin
) {
    float2 ndc = (screenCoord / ubo.windowSize) * 2.0 - 1.0;
    float4 nearClip = float4(ndc.x, ndc.y, 0.0, 1.0);
    float4 nearWorld = mul(ubo.invViewProj, nearClip);
    float3 nearWorldPos = nearWorld.xyz / nearWorld.w;
    return normalize(nearWorldPos - rayOrigin);
}

public float3 calculateCenterRayDirection(
    UniformBufferObject ubo,
    float3 rayOrigin
) {
    return calculateRayDirection(ubo, ubo.windowSize * 0.5, rayOrigin);
}
