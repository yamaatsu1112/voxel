// SVO Raycast - stack-based traversal specialized for 4x4x4 SV64 nodes.

implementing svo;

public struct RaycastInfo {
    uint nodeIndex;
    uint childIndex;
    float3 tMin;
    uint packedAncestorDepth;
};

public uint packAncestorDepth(int3 ancestorDepth) {
    uint3 packed = uint3(ancestorDepth + 1);
    return packed.x | (packed.y << 8u) | (packed.z << 16u);
}

public int unpackAncestorDepthComponent(uint packedAncestorDepth, uint axis) {
    return int((packedAncestorDepth >> (axis << 3u)) & 0xffu) - 1;
}

public uint updatePackedAncestorDepth(uint packedAncestorDepth, uint3 childBits, uint depth) {
    uint encodedDepth = depth + 1u;

    if (childBits.x != 0x3u) {
        packedAncestorDepth = (packedAncestorDepth & 0xffffff00u) | encodedDepth;
    }
    if (childBits.y != 0x3u) {
        packedAncestorDepth = (packedAncestorDepth & 0xffff00ffu) | (encodedDepth << 8u);
    }
    if (childBits.z != 0x3u) {
        packedAncestorDepth = (packedAncestorDepth & 0xff00ffffu) | (encodedDepth << 16u);
    }

    return packedAncestorDepth;
}

public uint3 childBits1FromIndex(uint childIndex) {
    return (childIndex >> uint3(1u, 3u, 5u)) & 1u;
}

public uint3 childBits2FromIndex(uint childIndex) {
    return (childIndex >> uint3(0u, 2u, 4u)) & 1u;
}

public uint buildChildIndex(uint3 bits1, uint3 bits2) {
    return
        (bits2.x << 0) | (bits1.x << 1) |
        (bits2.y << 2) | (bits1.y << 3) |
        (bits2.z << 4) | (bits1.z << 5);
}

public float3 childOffsetForDepth(uint childIndex, uint depth) {
    uint3 bits1 = childBits1FromIndex(childIndex);
    uint3 bits2 = childBits2FromIndex(childIndex);
    float nodeSize = nodeSizeForDepth(depth);

    return
        float3(bits1) * (nodeSize * 0.5) +
        float3(bits2) * (nodeSize * 0.25);
}

public uint3 calculateVoxelCoordFromRaycastInfo(
    RaycastInfo raycastInfo[MAX_DEPTH + 1],
    uint depth,
    float3 c0,
    float3 c1
) {
    uint3 voxelCoord = uint3(0, 0, 0);

    for (uint d = 0; d < depth; d++) {
        uint childIndex = raycastInfo[d].childIndex;
        uint shift = (MAX_DEPTH - d - 1u) * BRANCH_FACTOR_EXP + LEAF_VOXEL_COUNT_EXP;

        voxelCoord.x |= ((childIndex >> 0u) & 0x3u) << shift;
        voxelCoord.y |= ((childIndex >> 2u) & 0x3u) << shift;
        voxelCoord.z |= ((childIndex >> 4u) & 0x3u) << shift;
    }

    if (depth < MAX_DEPTH) {
        uint childIndex = raycastInfo[depth].childIndex;
        uint shift = (MAX_DEPTH - depth - 1u) * BRANCH_FACTOR_EXP + LEAF_VOXEL_COUNT_EXP;

        voxelCoord.x |= ((childIndex >> 0u) & 0x3u) << shift;
        voxelCoord.y |= ((childIndex >> 2u) & 0x3u) << shift;
        voxelCoord.z |= ((childIndex >> 4u) & 0x3u) << shift;

        float3 t0 = c0 * raycastInfo[depth].tMin + c1;
        float tcMin = max(t0.x, max(t0.y, t0.z));
        float3 tMin = raycastInfo[depth].tMin;

        for (uint d = depth + 1u; d < MAX_DEPTH; d++) {
            float nodeSize = nodeSizeForDepth(d);
            float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
            uint3 bits1 = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

            float3 nextDir = float3(bits1) * 2.0 - float3(1.0);
            float3 t2 = c0 * ((tMin + nodeSize * 0.5) + nextDir * (nodeSize * 0.25)) + c1;
            uint3 bits2 = select(tcMin < t2, uint3(0, 0, 0), uint3(1, 1, 1));
            uint3 bits = bits2 | (bits1 << 1u);

            shift = (MAX_DEPTH - d - 1u) * BRANCH_FACTOR_EXP + LEAF_VOXEL_COUNT_EXP;
            voxelCoord.x |= bits.x << shift;
            voxelCoord.y |= bits.y << shift;
            voxelCoord.z |= bits.z << shift;

            tMin += float3(bits1) * (nodeSize * 0.5);
            tMin += float3(bits2) * (nodeSize * 0.25);
        }

        float nodeSize = nodeSizeForDepth(MAX_DEPTH);
        float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
        uint3 bits1 = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

        float3 nextDir = float3(bits1) * 2.0 - float3(1.0);
        float3 t2 = c0 * ((tMin + nodeSize * 0.5) + nextDir * (nodeSize * 0.25)) + c1;
        uint3 bits2 = select(tcMin < t2, uint3(0, 0, 0), uint3(1, 1, 1));

        voxelCoord.x |= bits2.x | (bits1.x << 1u);
        voxelCoord.y |= bits2.y | (bits1.y << 1u);
        voxelCoord.z |= bits2.z | (bits1.z << 1u);
    } else {
        uint childIndex = raycastInfo[MAX_DEPTH].childIndex;

        voxelCoord.x |= (childIndex >> 0u) & 0x3u;
        voxelCoord.y |= (childIndex >> 2u) & 0x3u;
        voxelCoord.z |= (childIndex >> 4u) & 0x3u;
    }

    return voxelCoord;
}

public void goNextSibling(
    inout RaycastInfo raycastInfo[MAX_DEPTH + 1],
    uint depth,
    uint next,
    float3 nextDirection
) {
    raycastInfo[depth].tMin += nextDirection * childExtentForDepth(depth);

    uint bitOffset = next << 1u;
    uint currentBits = (raycastInfo[depth].childIndex >> bitOffset) & 0x3u;
    uint newBits = (currentBits + 1u) & 0x3u;
    raycastInfo[depth].childIndex =
        (raycastInfo[depth].childIndex & ~(0x3u << bitOffset)) |
        (newBits << bitOffset);
}

public struct StackRaycastImpl {}

public extension StackRaycastImpl {
    public static uint setupRaycastInfo(
        StructuredBuffer<SVONode> nodesBuffer,
        uint3 currentVoxel,
        inout RaycastInfo raycastInfo[MAX_DEPTH + 1],
        uint mirrorMask
    ) {
        uint nodeIndex = ROOT_NODE_INDEX;
        float3 tMin = float3(0.0);
        uint packedAncestorDepth = packAncestorDepth(int3(-1));

        for (uint d = 0u; d < MAX_DEPTH; d++) {
            raycastInfo[d].nodeIndex = nodeIndex;
            raycastInfo[d].packedAncestorDepth = packedAncestorDepth;

            uint childIndex = getChildIndex(currentVoxel.x, currentVoxel.y, currentVoxel.z, d);
            raycastInfo[d].childIndex = childIndex;
            tMin += childOffsetForDepth(childIndex, d);
            raycastInfo[d].tMin = tMin;

            uint mirroredChildIndex = childIndex ^ mirrorMask;
            if (!nodeHasChild(nodesBuffer, nodeIndex, mirroredChildIndex)) {
                return d;
            }

            nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, mirroredChildIndex);
            packedAncestorDepth = updatePackedAncestorDepth(
                packedAncestorDepth,
                (childIndex >> uint3(0u, 2u, 4u)) & 0x3u,
                d
            );
        }

        raycastInfo[MAX_DEPTH].nodeIndex = nodeIndex;
        raycastInfo[MAX_DEPTH].packedAncestorDepth = packedAncestorDepth;

        uint3 offset = currentVoxel & ((1u << LEAF_VOXEL_COUNT_EXP) - 1u);
        uint childIndex = buildLeafChildIndex(offset);
        raycastInfo[MAX_DEPTH].childIndex = childIndex;
        tMin += childOffsetForDepth(childIndex, MAX_DEPTH);
        raycastInfo[MAX_DEPTH].tMin = tMin;

        return MAX_DEPTH;
    }

    public static uint descendToLowestLevelNode(
        StructuredBuffer<SVONode> nodesBuffer,
        inout RaycastInfo raycastInfo[MAX_DEPTH + 1],
        uint depth,
        float3 c0,
        float3 c1,
        uint mirrorMask
    ) {
        uint nodeIndex = raycastInfo[depth].nodeIndex;
        uint childIndex = raycastInfo[depth].childIndex;
        uint mirroredChildIndex = childIndex ^ mirrorMask;

        if (!nodeHasChild(nodesBuffer, nodeIndex, mirroredChildIndex)) {
            return depth;
        }

        nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, mirroredChildIndex);
        uint packedAncestorDepth = updatePackedAncestorDepth(
            raycastInfo[depth].packedAncestorDepth,
            (childIndex >> uint3(0u, 2u, 4u)) & 0x3u,
            depth
        );

        float3 t0 = c0 * raycastInfo[depth].tMin + c1;
        float tcMin = max(t0.x, max(t0.y, t0.z));

        for (uint d = depth + 1u; d < MAX_DEPTH; d++) {
            raycastInfo[d].nodeIndex = nodeIndex;
            raycastInfo[d].packedAncestorDepth = packedAncestorDepth;

            float3 tMin = raycastInfo[d - 1u].tMin;
            float nodeSize = nodeSizeForDepth(d);
            float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
            uint3 bits1 = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

            float3 nextDir = float3(bits1) * 2.0 - float3(1.0);
            float3 t2 = c0 * ((tMin + nodeSize * 0.5) + nextDir * (nodeSize * 0.25)) + c1;
            uint3 bits2 = select(tcMin < t2, uint3(0, 0, 0), uint3(1, 1, 1));

            childIndex = buildChildIndex(bits1, bits2);
            raycastInfo[d].childIndex = childIndex;

            tMin += float3(bits1) * (nodeSize * 0.5);
            tMin += float3(bits2) * (nodeSize * 0.25);
            raycastInfo[d].tMin = tMin;

            mirroredChildIndex = childIndex ^ mirrorMask;
            if (!nodeHasChild(nodesBuffer, nodeIndex, mirroredChildIndex)) {
                return d;
            }

            nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, mirroredChildIndex);
            packedAncestorDepth = updatePackedAncestorDepth(
                packedAncestorDepth,
                bits2 | (bits1 << 1u),
                d
            );
        }

        raycastInfo[MAX_DEPTH].nodeIndex = nodeIndex;
        raycastInfo[MAX_DEPTH].packedAncestorDepth = packedAncestorDepth;

        float3 tMin = raycastInfo[MAX_DEPTH - 1u].tMin;
        float nodeSize = nodeSizeForDepth(MAX_DEPTH);
        float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
        uint3 bits1 = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

        float3 nextDir = float3(bits1) * 2.0 - float3(1.0);
        float3 t2 = c0 * ((tMin + nodeSize * 0.5) + nextDir * (nodeSize * 0.25)) + c1;
        uint3 bits2 = select(tcMin < t2, uint3(0, 0, 0), uint3(1, 1, 1));

        childIndex = buildChildIndex(bits1, bits2);
        raycastInfo[MAX_DEPTH].childIndex = childIndex;

        tMin += float3(bits1) * (nodeSize * 0.5);
        tMin += float3(bits2) * (nodeSize * 0.25);
        raycastInfo[MAX_DEPTH].tMin = tMin;

        return MAX_DEPTH;
    }

    public static RaycastResult raycast(
        StructuredBuffer<SVONode> nodesBuffer,
        StructuredBuffer<SVOLeaf> leavesBuffer,
        float3 rayOrigin,
        float3 rayDirection
    ) {
        RaycastResult noHitResult = makeNoHitRaycastResult();
        RaycastSetup setup = prepareRaycastSetup(rayOrigin, rayDirection);
        if (!setup.valid) {
            return noHitResult;
        }

        RaycastInfo raycastInfo[MAX_DEPTH + 1];
        uint3 currentVoxel = normalizedToVoxelCoord(setup.rayOriginNormalized);
        uint depth = StackRaycastImpl.setupRaycastInfo(
            nodesBuffer,
            currentVoxel,
            raycastInfo,
            setup.mirrorMask
        );

        Face face = setup.face;

        for (uint i = 0u; i < WORLD_VOXEL_COUNT; i++) {
            float3 t0 = setup.c0 * raycastInfo[depth].tMin + setup.c1;
            float tcMin = max(t0.x, max(t0.y, t0.z));
            uint nodeIndex = raycastInfo[depth].nodeIndex;
            uint childIndex = raycastInfo[depth].childIndex;

            if (depth == MAX_DEPTH) {
                if (leavesBuffer[nodeIndex].getVoxel(childIndex ^ setup.mirrorMask)) {
                    RaycastResult result;
                    result.face = unmirrorHitFace(face, setup.negDirection);
                    result.tc_min = tcMin + setup.tOffset;
                    result.voxelCoord = unmirrorVoxelCoord(
                        calculateVoxelCoordFromRaycastInfo(
                            raycastInfo,
                            depth,
                            setup.c0,
                            setup.c1
                        ),
                        setup.posDirection,
                        setup.negDirection
                    );
                    return result;
                }
            } else if (nodeIsFilled(nodesBuffer, nodeIndex, childIndex ^ setup.mirrorMask)) {
                RaycastResult result;
                result.face = unmirrorHitFace(face, setup.negDirection);
                result.tc_min = tcMin + setup.tOffset;
                result.voxelCoord = unmirrorVoxelCoord(
                    calculateVoxelCoordFromRaycastInfo(
                        raycastInfo,
                        depth,
                        setup.c0,
                        setup.c1
                    ),
                    setup.posDirection,
                    setup.negDirection
                );
                return result;
            }

            float3 t = setup.c0 * (raycastInfo[depth].tMin + childExtentForDepth(depth)) + setup.c1;
            float tcMax = min(t.x, min(t.y, t.z));

            uint next;
            float3 nextDirection;

            if (tcMax == t.x) {
                next = 0u;
                nextDirection = float3(1.0, 0.0, 0.0);
                face = Face::X_NEG;
            } else if (tcMax == t.y) {
                next = 1u;
                nextDirection = float3(0.0, 1.0, 0.0);
                face = Face::Y_NEG;
            } else {
                next = 2u;
                nextDirection = float3(0.0, 0.0, 1.0);
                face = Face::Z_NEG;
            }

            if (((childIndex >> (next << 1u)) & 0x3u) == 0x3u) {
                int ancestorDepth = unpackAncestorDepthComponent(
                    raycastInfo[depth].packedAncestorDepth,
                    next
                );
                if (ancestorDepth <= -1) {
                    return noHitResult;
                }
                depth = uint(ancestorDepth);
            }

            goNextSibling(raycastInfo, depth, next, nextDirection);

            if (depth < MAX_DEPTH) {
                depth = StackRaycastImpl.descendToLowestLevelNode(
                    nodesBuffer,
                    raycastInfo,
                    depth,
                    setup.c0,
                    setup.c1,
                    setup.mirrorMask
                );
            }
        }

        return noHitResult;
    }

    public static bool raycastAnyHit(
        StructuredBuffer<SVONode> nodesBuffer,
        StructuredBuffer<SVOLeaf> leavesBuffer,
        float3 rayOrigin,
        float3 rayDirection
    ) {
        RaycastSetup setup = prepareRaycastSetup(rayOrigin, rayDirection);
        if (!setup.valid) {
            return false;
        }

        RaycastInfo raycastInfo[MAX_DEPTH + 1];
        uint3 currentVoxel = normalizedToVoxelCoord(setup.rayOriginNormalized);
        uint depth = StackRaycastImpl.setupRaycastInfo(
            nodesBuffer,
            currentVoxel,
            raycastInfo,
            setup.mirrorMask
        );

        for (uint i = 0u; i < WORLD_VOXEL_COUNT; i++) {
            uint nodeIndex = raycastInfo[depth].nodeIndex;
            uint childIndex = raycastInfo[depth].childIndex;

            if (depth == MAX_DEPTH) {
                if (leavesBuffer[nodeIndex].getVoxel(childIndex ^ setup.mirrorMask)) {
                    return true;
                }
            } else if (nodeIsFilled(nodesBuffer, nodeIndex, childIndex ^ setup.mirrorMask)) {
                return true;
            }

            float3 t = setup.c0 * (raycastInfo[depth].tMin + childExtentForDepth(depth)) + setup.c1;
            float tcMax = min(t.x, min(t.y, t.z));

            uint next;
            float3 nextDirection;

            if (tcMax == t.x) {
                next = 0u;
                nextDirection = float3(1.0, 0.0, 0.0);
            } else if (tcMax == t.y) {
                next = 1u;
                nextDirection = float3(0.0, 1.0, 0.0);
            } else {
                next = 2u;
                nextDirection = float3(0.0, 0.0, 1.0);
            }

            if (((childIndex >> (next << 1u)) & 0x3u) == 0x3u) {
                int ancestorDepth = unpackAncestorDepthComponent(
                    raycastInfo[depth].packedAncestorDepth,
                    next
                );
                if (ancestorDepth <= -1) {
                    return false;
                }
                depth = uint(ancestorDepth);
            }

            goNextSibling(raycastInfo, depth, next, nextDirection);

            if (depth < MAX_DEPTH) {
                depth = StackRaycastImpl.descendToLowestLevelNode(
                    nodesBuffer,
                    raycastInfo,
                    depth,
                    setup.c0,
                    setup.c1,
                    setup.mirrorMask
                );
            }
        }

        return false;
    }
}
