// SVO Raycast - Stack-based traversal implementation.

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

public uint updatePackedAncestorDepth(uint packedAncestorDepth, uint3 bits, uint depth) {
    uint encodedDepth = depth + 1u;

    if (bits.x == 0u) {
        packedAncestorDepth = (packedAncestorDepth & 0xffffff00u) | encodedDepth;
    }
    if (bits.y == 0u) {
        packedAncestorDepth = (packedAncestorDepth & 0xffff00ffu) | (encodedDepth << 8u);
    }
    if (bits.z == 0u) {
        packedAncestorDepth = (packedAncestorDepth & 0xff00ffffu) | (encodedDepth << 16u);
    }

    return packedAncestorDepth;
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
        uint shift = MAX_DEPTH - d - 1 + LEAF_VOXEL_COUNT_EXP;

        voxelCoord.x |= ((childIndex >> 0) & 1u) << shift;
        voxelCoord.y |= ((childIndex >> 1) & 1u) << shift;
        voxelCoord.z |= ((childIndex >> 2) & 1u) << shift;
    }

    if (depth < MAX_DEPTH) {
        uint childIndex = raycastInfo[depth].childIndex;
        uint shift = MAX_DEPTH - depth - 1 + LEAF_VOXEL_COUNT_EXP;

        voxelCoord.x |= ((childIndex >> 0) & 1u) << shift;
        voxelCoord.y |= ((childIndex >> 1) & 1u) << shift;
        voxelCoord.z |= ((childIndex >> 2) & 1u) << shift;

        float3 t0 = c0 * raycastInfo[depth].tMin + c1;
        float tcMin = max(t0.x, max(t0.y, t0.z));

        float3 tMin = raycastInfo[depth].tMin;
        for (uint d = depth + 1; d < MAX_DEPTH; d++) {
            float nodeSize = nodeSizeForDepth(d);
            float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
            uint3 bits = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

            shift = MAX_DEPTH - d - 1 + LEAF_VOXEL_COUNT_EXP;
            voxelCoord.x |= bits.x << shift;
            voxelCoord.y |= bits.y << shift;
            voxelCoord.z |= bits.z << shift;

            tMin += float3(bits) * (nodeSize * 0.5);
        }

        float nodeSize = nodeSizeForDepth(MAX_DEPTH);
        float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
        uint3 bits1 = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

        float3 nextDir = float3(bits1) * 2.0 - float3(1.0);
        float3 t2 = c0 * ((tMin + nodeSize * 0.5) + nextDir * (nodeSize * 0.25)) + c1;
        uint3 bits2 = select(tcMin < t2, uint3(0, 0, 0), uint3(1, 1, 1));

        voxelCoord.x |= (bits2.x << 0) | (bits1.x << 1);
        voxelCoord.y |= (bits2.y << 0) | (bits1.y << 1);
        voxelCoord.z |= (bits2.z << 0) | (bits1.z << 1);
    } else {
        uint childIndex = raycastInfo[MAX_DEPTH].childIndex;

        voxelCoord.x |= ((childIndex >> 0) & 1u) | (((childIndex >> 1) & 1u) << 1);
        voxelCoord.y |= ((childIndex >> 2) & 1u) | (((childIndex >> 3) & 1u) << 1);
        voxelCoord.z |= ((childIndex >> 4) & 1u) | (((childIndex >> 5) & 1u) << 1);
    }

    return voxelCoord;
}

public void goNextSiblingNode(
    inout RaycastInfo raycastInfo[MAX_DEPTH + 1],
    uint depth,
    uint next,
    float3 nextDirection
) {
    float nodeSize = nodeSizeForDepth(depth);
    raycastInfo[depth].tMin += nextDirection * (nodeSize * 0.5);
    raycastInfo[depth].childIndex = raycastInfo[depth].childIndex ^ (1u << next);
}

public void goNextSiblingVoxel(
    inout RaycastInfo raycastInfo[MAX_DEPTH + 1],
    uint depth,
    uint next,
    float3 nextDirection
) {
    float nodeSize = nodeSizeForDepth(depth);
    raycastInfo[depth].tMin += nextDirection * (nodeSize * LEAF_VOXEL_COUNT_INV);

    uint bitOffset = next * 2;
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

        for (uint d = 0; d < MAX_DEPTH; d++) {
            raycastInfo[d].nodeIndex = nodeIndex;
            raycastInfo[d].packedAncestorDepth = packedAncestorDepth;

            float nodeSize = nodeSizeForDepth(d);
            uint3 bits = (currentVoxel >> (MAX_DEPTH - d - 1 + LEAF_VOXEL_COUNT_EXP)) & 1u;
            uint childIndex = (bits.x << 0) | (bits.y << 1) | (bits.z << 2);

            raycastInfo[d].childIndex = childIndex;
            tMin += float3(bits) * (nodeSize * 0.5);
            raycastInfo[d].tMin = tMin;

            uint mirroredChildIndex = childIndex ^ mirrorMask;
            if (!nodeHasChild(nodesBuffer, nodeIndex, mirroredChildIndex)) {
                return d;
            }

            nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, mirroredChildIndex);
            packedAncestorDepth =
                updatePackedAncestorDepth(raycastInfo[d].packedAncestorDepth, bits, d);
        }

        raycastInfo[MAX_DEPTH].nodeIndex = nodeIndex;
        raycastInfo[MAX_DEPTH].packedAncestorDepth = packedAncestorDepth;

        float nodeSize = nodeSizeForDepth(MAX_DEPTH);
        uint3 offset = currentVoxel & ((1u << LEAF_VOXEL_COUNT_EXP) - 1u);
        raycastInfo[MAX_DEPTH].childIndex = buildLeafChildIndex(offset);

        uint3 bits1 = (offset >> 1) & 1u;
        uint3 bits2 = offset & 1u;

        tMin += float3(bits1) * (nodeSize * 0.5);
        tMin += float3(bits2) * (nodeSize * 0.25);
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
        uint packedAncestorDepth = raycastInfo[depth].packedAncestorDepth;
        uint3 bits = (childIndex >> uint3(0, 1, 2)) & 1u;
        packedAncestorDepth = updatePackedAncestorDepth(packedAncestorDepth, bits, depth);

        float3 t0 = c0 * raycastInfo[depth].tMin + c1;
        float tcMin = max(t0.x, max(t0.y, t0.z));

        for (uint d = depth + 1; d < MAX_DEPTH; d++) {
            raycastInfo[d].nodeIndex = nodeIndex;
            raycastInfo[d].packedAncestorDepth = packedAncestorDepth;

            float3 tMin = raycastInfo[d - 1].tMin;
            float nodeSize = nodeSizeForDepth(d);
            float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
            bits = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

            childIndex = (bits.x << 0) | (bits.y << 1) | (bits.z << 2);
            raycastInfo[d].childIndex = childIndex;

            tMin += float3(bits) * (nodeSize * 0.5);
            raycastInfo[d].tMin = tMin;

            mirroredChildIndex = childIndex ^ mirrorMask;
            if (!nodeHasChild(nodesBuffer, nodeIndex, mirroredChildIndex)) {
                return d;
            }

            nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, mirroredChildIndex);
            packedAncestorDepth =
                updatePackedAncestorDepth(raycastInfo[d].packedAncestorDepth, bits, d);
        }

        raycastInfo[MAX_DEPTH].nodeIndex = nodeIndex;
        raycastInfo[MAX_DEPTH].packedAncestorDepth = packedAncestorDepth;

        float3 tMin = raycastInfo[MAX_DEPTH - 1].tMin;
        float nodeSize = nodeSizeForDepth(MAX_DEPTH);

        float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
        uint3 bits1 = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

        float3 nextDir = float3(bits1) * 2.0 - float3(1.0);
        float3 t2 = c0 * ((tMin + nodeSize * 0.5) + nextDir * (nodeSize * 0.25)) + c1;
        uint3 bits2 = select(tcMin < t2, uint3(0, 0, 0), uint3(1, 1, 1));

        raycastInfo[MAX_DEPTH].childIndex =
            (bits2.x << 0) | (bits1.x << 1) |
            (bits2.y << 2) | (bits1.y << 3) |
            (bits2.z << 4) | (bits1.z << 5);

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

        for (uint i = 0; i < WORLD_VOXEL_COUNT; i++) {
            float3 t0 = setup.c0 * raycastInfo[depth].tMin + setup.c1;
            float tcMin = max(t0.x, max(t0.y, t0.z));
            uint nodeIndex = raycastInfo[depth].nodeIndex;

            if (depth == MAX_DEPTH) {
                SVOLeaf leaf = leavesBuffer[nodeIndex];
                uint childIndex = raycastInfo[MAX_DEPTH].childIndex ^ setup.mirrorMaskLeaf;

                if (leaf.getVoxel(childIndex)) {
                    RaycastResult result;
                    result.face = unmirrorHitFace(face, setup.negDirection);
                    result.tc_min = tcMin + setup.tOffset;

                    uint3 voxelCoord = calculateVoxelCoordFromRaycastInfo(
                        raycastInfo,
                        depth,
                        setup.c0,
                        setup.c1
                    );
                    result.voxelCoord = unmirrorVoxelCoord(
                        voxelCoord,
                        setup.posDirection,
                        setup.negDirection
                    );
                    return result;
                }
            } else {
                uint childIndex = raycastInfo[depth].childIndex;

                if (nodeIsFilled(nodesBuffer, nodeIndex, childIndex ^ setup.mirrorMask)) {
                    RaycastResult result;
                    result.face = unmirrorHitFace(face, setup.negDirection);
                    result.tc_min = tcMin + setup.tOffset;

                    uint3 voxelCoord = calculateVoxelCoordFromRaycastInfo(
                        raycastInfo,
                        depth,
                        setup.c0,
                        setup.c1
                    );
                    result.voxelCoord = unmirrorVoxelCoord(
                        voxelCoord,
                        setup.posDirection,
                        setup.negDirection
                    );
                    return result;
                }
            }

            uint childIndex = raycastInfo[depth].childIndex;
            float3 t = setup.c0 *
                       (raycastInfo[depth].tMin + childExtentForDepth(depth)) +
                       setup.c1;
            float tcMax = min(t.x, min(t.y, t.z));

            uint next;
            float3 nextDirection;

            if (tcMax == t.x) {
                next = 0;
                nextDirection = float3(1.0, 0.0, 0.0);
                face = Face::X_NEG;
            } else if (tcMax == t.y) {
                next = 1;
                nextDirection = float3(0.0, 1.0, 0.0);
                face = Face::Y_NEG;
            } else {
                next = 2;
                nextDirection = float3(0.0, 0.0, 1.0);
                face = Face::Z_NEG;
            }

            if (depth == MAX_DEPTH) {
                if (((childIndex >> (next << 1)) & 0x3u) == 0x3u) {
                    int ancestorDepth = unpackAncestorDepthComponent(
                        raycastInfo[depth].packedAncestorDepth,
                        next
                    );
                    if (ancestorDepth <= -1) {
                        return noHitResult;
                    }
                    depth = uint(ancestorDepth);
                } else {
                    goNextSiblingVoxel(raycastInfo, depth, next, nextDirection);
                    continue;
                }
            } else if (((childIndex >> next) & 1u) != 0u) {
                int ancestorDepth = unpackAncestorDepthComponent(
                    raycastInfo[depth].packedAncestorDepth,
                    next
                );
                if (ancestorDepth <= -1) {
                    return noHitResult;
                }
                depth = uint(ancestorDepth);
            }

            goNextSiblingNode(raycastInfo, depth, next, nextDirection);
            depth = StackRaycastImpl.descendToLowestLevelNode(
                nodesBuffer,
                raycastInfo,
                depth,
                setup.c0,
                setup.c1,
                setup.mirrorMask
            );
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

        for (uint i = 0; i < WORLD_VOXEL_COUNT; i++) {
            uint nodeIndex = raycastInfo[depth].nodeIndex;

            if (depth == MAX_DEPTH) {
                SVOLeaf leaf = leavesBuffer[nodeIndex];
                uint childIndex = raycastInfo[MAX_DEPTH].childIndex ^ setup.mirrorMaskLeaf;
                if (leaf.getVoxel(childIndex)) {
                    return true;
                }
            } else {
                uint childIndex = raycastInfo[depth].childIndex;
                if (nodeIsFilled(nodesBuffer, nodeIndex, childIndex ^ setup.mirrorMask)) {
                    return true;
                }
            }

            uint childIndex = raycastInfo[depth].childIndex;
            float3 t = setup.c0 *
                       (raycastInfo[depth].tMin + childExtentForDepth(depth)) +
                       setup.c1;
            float tcMax = min(t.x, min(t.y, t.z));

            uint next;
            float3 nextDirection;

            if (tcMax == t.x) {
                next = 0;
                nextDirection = float3(1.0, 0.0, 0.0);
            } else if (tcMax == t.y) {
                next = 1;
                nextDirection = float3(0.0, 1.0, 0.0);
            } else {
                next = 2;
                nextDirection = float3(0.0, 0.0, 1.0);
            }

            if (depth == MAX_DEPTH) {
                if (((childIndex >> (next << 1)) & 0x3u) == 0x3u) {
                    int ancestorDepth = unpackAncestorDepthComponent(
                        raycastInfo[depth].packedAncestorDepth,
                        next
                    );
                    if (ancestorDepth <= -1) {
                        return false;
                    }
                    depth = uint(ancestorDepth);
                } else {
                    goNextSiblingVoxel(raycastInfo, depth, next, nextDirection);
                    continue;
                }
            } else if (((childIndex >> next) & 1u) != 0u) {
                int ancestorDepth = unpackAncestorDepthComponent(
                    raycastInfo[depth].packedAncestorDepth,
                    next
                );
                if (ancestorDepth <= -1) {
                    return false;
                }
                depth = uint(ancestorDepth);
            }

            goNextSiblingNode(raycastInfo, depth, next, nextDirection);
            depth = StackRaycastImpl.descendToLowestLevelNode(
                nodesBuffer,
                raycastInfo,
                depth,
                setup.c0,
                setup.c1,
                setup.mirrorMask
            );
        }

        return false;
    }
}
