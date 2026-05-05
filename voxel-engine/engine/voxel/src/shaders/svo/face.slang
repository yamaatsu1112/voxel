// SVO Face - Face Direction Enum
// This module contains the Face enum for surface direction representation.

implementing svo;

// ============================================================================
// Face Direction Enum
// ============================================================================

public enum Face : uint {
    X_POS = 0,
    X_NEG = 1,
    Y_POS = 2,
    Y_NEG = 3,
    Z_POS = 4,
    Z_NEG = 5,
    NO_HIT = 6
};

// Raycast result
public struct RaycastResult {
    public Face face;
    public float tc_min;
    public uint3 voxelCoord; // Voxel coordinates at hit point
};
