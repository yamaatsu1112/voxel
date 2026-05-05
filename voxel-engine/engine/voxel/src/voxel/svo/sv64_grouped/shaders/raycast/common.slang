// SVO Raycast - Shared helpers for selectable traversal implementations.

implementing svo;

public static const float RAYCAST_ENTRY_EPS = 1e-4;

public struct RaycastSetup {
    public int3 posDirection;
    public int3 negDirection;
    public uint mirrorMask;
    public float3 rayDirection;
    public float3 rayOriginNormalized;
    public float3 c0;
    public float3 c1;
    public float tOffset;
    public Face face;
    public bool valid;
};

public RaycastResult makeNoHitRaycastResult() {
    RaycastResult result;
    result.face = Face::NO_HIT;
    result.tc_min = -1.0;
    result.voxelCoord = uint3(~0u, ~0u, ~0u);
    return result;
}

public Face unmirrorHitFace(Face face, int3 negDirection) {
    if (face == Face::X_NEG && negDirection.x == 1) return Face::X_POS;
    if (face == Face::Y_NEG && negDirection.y == 1) return Face::Y_POS;
    if (face == Face::Z_NEG && negDirection.z == 1) return Face::Z_POS;
    return face;
}

public uint3 unmirrorVoxelCoord(uint3 voxelCoord, int3 posDirection, int3 negDirection) {
    return uint3(negDirection) * (uint3(WORLD_VOXEL_COUNT - 1u) - voxelCoord) +
           uint3(posDirection) * voxelCoord;
}

public uint buildLeafChildIndex(uint3 offset) {
    return offset.x | (offset.y << LEAF_VOXEL_COUNT_EXP) |
           (offset.z << (LEAF_VOXEL_COUNT_EXP << 1));
}

public uint3 normalizedToVoxelCoord(float3 normalizedPosition) {
    uint3 voxelCoord = uint3(floor(
        normalizedPosition *
        float3(float(WORLD_VOXEL_COUNT))
    ));

    return min(
        voxelCoord,
        uint3(WORLD_VOXEL_COUNT - 1u)
    );
}

public RaycastSetup prepareRaycastSetup(float3 rayOrigin, float3 rayDirection) {
    RaycastSetup setup;

    setup.posDirection = select(rayDirection >= 0.0, int3(1, 1, 1), int3(0, 0, 0));
    setup.negDirection = int3(1, 1, 1) - setup.posDirection;

    setup.mirrorMask = makeMirrorMask2BitsPerAxis(setup.negDirection);

    setup.rayDirection = abs(rayDirection);
    setup.tOffset = 0.0;
    setup.face = Face::X_POS;
    setup.valid = false;

    float3 worldHalfSize = float3((WORLD_VOXEL_COUNT >> VOXEL_SIZE_NEG_EXP) >> 1);
    float worldScale = float(WORLD_VOXEL_COUNT >> VOXEL_SIZE_NEG_EXP);
    float3 rayOriginNormalized = (rayOrigin + worldHalfSize) / worldScale;

    rayOriginNormalized =
        float3(setup.negDirection) * (1.0 - rayOriginNormalized) +
        float3(setup.posDirection) * rayOriginNormalized;

    setup.rayOriginNormalized = rayOriginNormalized;
    setup.c0 = float3(1.0) / setup.rayDirection;
    setup.c1 = (-rayOriginNormalized) * setup.c0;

    float3 tNear = setup.c1;
    float3 tFar = setup.c0 + setup.c1;
    float tEntry = max(tNear.x, max(tNear.y, tNear.z));
    float tExit = min(tFar.x, min(tFar.y, tFar.z));

    if (tEntry >= tExit || tExit < 0.0) {
        return setup;
    }

    setup.valid = true;

    if (tEntry > 0.0) {
        setup.tOffset = tEntry;

        if (tEntry == tNear.x) {
            setup.face = Face::X_NEG;
        } else if (tEntry == tNear.y) {
            setup.face = Face::Y_NEG;
        } else {
            setup.face = Face::Z_NEG;
        }

        setup.rayOriginNormalized = clamp(
            rayOriginNormalized + setup.rayDirection * tEntry,
            float3(RAYCAST_ENTRY_EPS),
            float3(1.0 - RAYCAST_ENTRY_EPS)
        );
        setup.c1 = (-setup.rayOriginNormalized) * setup.c0;
    }

    return setup;
}
