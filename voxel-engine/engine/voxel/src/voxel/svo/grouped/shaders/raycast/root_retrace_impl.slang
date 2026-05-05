// SVO Raycast - Root retrace traversal implementation.

implementing svo;

public uint cellVoxelExtentForDepth(uint depth) {
    if (depth < MAX_DEPTH) {
        return 1u << (MAX_DEPTH - depth - 1u + LEAF_VOXEL_COUNT_EXP);
    }

    return 1u;
}

public uint depthToVoxelShift(uint depth) {
    if (depth < MAX_DEPTH) {
        return MAX_DEPTH - depth - 1u + LEAF_VOXEL_COUNT_EXP;
    }

    return 0u;
}

public uint3 alignVoxelCoordToDepth(uint3 voxelCoord, uint depth) {
    if (depth == MAX_DEPTH) {
        return voxelCoord;
    }

    uint shift = depthToVoxelShift(depth);
    uint mask = ~((1u << shift) - 1u);
    voxelCoord.x &= mask;
    voxelCoord.y &= mask;
    voxelCoord.z &= mask;
    return voxelCoord;
}

public uint3 selectNodeChildBits(float tcMin, float3 tMin, uint depth, float3 c0, float3 c1) {
    float nodeSize = nodeSizeForDepth(depth);
    float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
    return select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));
}

public int findAncestorDepthForAxis(uint3 voxelCoord, uint axis, uint depth) {
    uint shiftStart = LEAF_VOXEL_COUNT_EXP;
    if (depth < MAX_DEPTH) {
        shiftStart = depthToVoxelShift(depth) + 1u;
    }

    uint axisCoord = voxelCoord.x;
    if (axis == 1u) {
        axisCoord = voxelCoord.y;
    } else if (axis == 2u) {
        axisCoord = voxelCoord.z;
    }

    for (uint shift = shiftStart; shift < MAX_DEPTH + LEAF_VOXEL_COUNT_EXP; shift++) {
        if (((axisCoord >> shift) & 1u) == 0u) {
            return int(MAX_DEPTH + LEAF_VOXEL_COUNT_EXP - shift - 1u);
        }
    }

    return -1;
}

public uint3 calculateVoxelCoordFromCurrentCell(
    uint depth,
    uint3 voxelCoord,
    float3 tMin,
    float tcMin,
    float3 c0,
    float3 c1
) {
    if (depth == MAX_DEPTH) {
        return voxelCoord;
    }

    for (uint d = depth + 1; d < MAX_DEPTH; d++) {
        uint3 bits = selectNodeChildBits(tcMin, tMin, d, c0, c1);
        uint shift = depthToVoxelShift(d);

        voxelCoord.x |= bits.x << shift;
        voxelCoord.y |= bits.y << shift;
        voxelCoord.z |= bits.z << shift;

        float nodeSize = nodeSizeForDepth(d);
        tMin += float3(bits) * (nodeSize * 0.5);
    }

    float nodeSize = nodeSizeForDepth(MAX_DEPTH);
    float3 t1 = c0 * (tMin + nodeSize * 0.5) + c1;
    uint3 bits1 = select(tcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

    float3 nextDir = float3(bits1) * 2.0 - float3(1.0);
    float3 t2 = c0 * ((tMin + nodeSize * 0.5) + nextDir * (nodeSize * 0.25)) + c1;
    uint3 bits2 = select(tcMin < t2, uint3(0, 0, 0), uint3(1, 1, 1));

    voxelCoord.x |= (bits1.x << 1) | bits2.x;
    voxelCoord.y |= (bits1.y << 1) | bits2.y;
    voxelCoord.z |= (bits1.z << 1) | bits2.z;

    return voxelCoord;
}

public struct RootRetraceRaycastImpl {}

public extension RootRetraceRaycastImpl {
    public static uint retraceFromRoot(
        StructuredBuffer<SVONode> nodesBuffer,
        uint3 fixedVoxelCoord,
        uint fixedDepth,
        float entryTcMin,
        float3 c0,
        float3 c1,
        uint mirrorMask,
        out uint3 currentVoxelCoord,
        out float3 currentTMin,
        out uint currentNodeIndex,
        out uint currentChildIndex
    ) {
        currentVoxelCoord = uint3(0, 0, 0);
        currentTMin = float3(0.0);
        currentNodeIndex = ROOT_NODE_INDEX;
        currentChildIndex = 0u;

        for (uint d = 0; d < MAX_DEPTH; d++) {
            uint3 bits;
            if (d <= fixedDepth) {
                bits = (fixedVoxelCoord >> depthToVoxelShift(d)) & 1u;
            } else {
                bits = selectNodeChildBits(entryTcMin, currentTMin, d, c0, c1);
            }

            currentChildIndex = (bits.x << 0) | (bits.y << 1) | (bits.z << 2);

            uint shift = depthToVoxelShift(d);
            currentVoxelCoord.x |= bits.x << shift;
            currentVoxelCoord.y |= bits.y << shift;
            currentVoxelCoord.z |= bits.z << shift;

            float nodeSize = nodeSizeForDepth(d);
            currentTMin += float3(bits) * (nodeSize * 0.5);

            uint mirroredChildIndex = currentChildIndex ^ mirrorMask;
            if (!nodeHasChild(nodesBuffer, currentNodeIndex, mirroredChildIndex)) {
                return d;
            }

            currentNodeIndex = nodeChildIndex(nodesBuffer, currentNodeIndex, mirroredChildIndex);
        }

        if (fixedDepth == MAX_DEPTH) {
            uint3 offset = fixedVoxelCoord & (LEAF_VOXEL_COUNT - 1u);
            uint3 bits1 = (offset >> 1u) & 1u;
            uint3 bits2 = offset & 1u;

            currentChildIndex = buildLeafChildIndex(offset);
            currentVoxelCoord = fixedVoxelCoord;

            float nodeSize = nodeSizeForDepth(MAX_DEPTH);
            currentTMin += float3(bits1) * (nodeSize * 0.5);
            currentTMin += float3(bits2) * (nodeSize * 0.25);
            return MAX_DEPTH;
        }

        float nodeSize = nodeSizeForDepth(MAX_DEPTH);
        float3 t1 = c0 * (currentTMin + nodeSize * 0.5) + c1;
        uint3 bits1 = select(entryTcMin < t1, uint3(0, 0, 0), uint3(1, 1, 1));

        float3 nextDir = float3(bits1) * 2.0 - float3(1.0);
        float3 t2 = c0 * ((currentTMin + nodeSize * 0.5) + nextDir * (nodeSize * 0.25)) + c1;
        uint3 bits2 = select(entryTcMin < t2, uint3(0, 0, 0), uint3(1, 1, 1));

        currentChildIndex =
            (bits2.x << 0) | (bits1.x << 1) |
            (bits2.y << 2) | (bits1.y << 3) |
            (bits2.z << 4) | (bits1.z << 5);

        currentVoxelCoord.x |= (bits1.x << 1) | bits2.x;
        currentVoxelCoord.y |= (bits1.y << 1) | bits2.y;
        currentVoxelCoord.z |= (bits1.z << 1) | bits2.z;

        currentTMin += float3(bits1) * (nodeSize * 0.5);
        currentTMin += float3(bits2) * (nodeSize * 0.25);

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

        uint3 currentVoxelCoord;
        float3 currentTMin;
        uint currentNodeIndex;
        uint currentChildIndex;
        uint depth = RootRetraceRaycastImpl.retraceFromRoot(
            nodesBuffer,
            normalizedToVoxelCoord(setup.rayOriginNormalized),
            MAX_DEPTH,
            0.0,
            setup.c0,
            setup.c1,
            setup.mirrorMask,
            currentVoxelCoord,
            currentTMin,
            currentNodeIndex,
            currentChildIndex
        );

        Face face = setup.face;

        for (uint i = 0; i < WORLD_VOXEL_COUNT; i++) {
            float3 t0 = setup.c0 * currentTMin + setup.c1;
            float tcMin = max(t0.x, max(t0.y, t0.z));

            if (depth == MAX_DEPTH) {
                SVOLeaf leaf = leavesBuffer[currentNodeIndex];
                uint childIndex = currentChildIndex ^ setup.mirrorMaskLeaf;

                if (leaf.getVoxel(childIndex)) {
                    RaycastResult result;
                    result.face = unmirrorHitFace(face, setup.negDirection);
                    result.tc_min = tcMin + setup.tOffset;
                    result.voxelCoord = unmirrorVoxelCoord(
                        currentVoxelCoord,
                        setup.posDirection,
                        setup.negDirection
                    );
                    return result;
                }
            } else {
                if (nodeIsFilled(nodesBuffer, currentNodeIndex, currentChildIndex ^ setup.mirrorMask)) {
                    RaycastResult result;
                    result.face = unmirrorHitFace(face, setup.negDirection);
                    result.tc_min = tcMin + setup.tOffset;

                    uint3 voxelCoord = calculateVoxelCoordFromCurrentCell(
                        depth,
                        currentVoxelCoord,
                        currentTMin,
                        tcMin,
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

            float3 t = setup.c0 * (currentTMin + childExtentForDepth(depth)) + setup.c1;
            float tcMax = min(t.x, min(t.y, t.z));

            uint next;
            Face nextFace;

            if (tcMax == t.x) {
                next = 0u;
                nextFace = Face::X_NEG;
            } else if (tcMax == t.y) {
                next = 1u;
                nextFace = Face::Y_NEG;
            } else {
                next = 2u;
                nextFace = Face::Z_NEG;
            }

            uint nextFixedDepth = depth;
            if (depth == MAX_DEPTH) {
                if (((currentChildIndex >> (next << 1)) & 0x3u) == 0x3u) {
                    int ancestorDepth = findAncestorDepthForAxis(currentVoxelCoord, next, depth);
                    if (ancestorDepth <= -1) {
                        return noHitResult;
                    }
                    nextFixedDepth = uint(ancestorDepth);
                }
            } else if (((currentChildIndex >> next) & 1u) != 0u) {
                int ancestorDepth = findAncestorDepthForAxis(currentVoxelCoord, next, depth);
                if (ancestorDepth <= -1) {
                    return noHitResult;
                }
                nextFixedDepth = uint(ancestorDepth);
            }

            uint3 nextFixedVoxelCoord = currentVoxelCoord;
            uint cellExtent = cellVoxelExtentForDepth(depth);
            if (next == 0u) {
                nextFixedVoxelCoord.x += cellExtent;
            } else if (next == 1u) {
                nextFixedVoxelCoord.y += cellExtent;
            } else {
                nextFixedVoxelCoord.z += cellExtent;
            }

            nextFixedVoxelCoord = alignVoxelCoordToDepth(nextFixedVoxelCoord, nextFixedDepth);

            depth = RootRetraceRaycastImpl.retraceFromRoot(
                nodesBuffer,
                nextFixedVoxelCoord,
                nextFixedDepth,
                tcMax,
                setup.c0,
                setup.c1,
                setup.mirrorMask,
                currentVoxelCoord,
                currentTMin,
                currentNodeIndex,
                currentChildIndex
            );
            face = nextFace;
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

        uint3 currentVoxelCoord;
        float3 currentTMin;
        uint currentNodeIndex;
        uint currentChildIndex;
        uint depth = RootRetraceRaycastImpl.retraceFromRoot(
            nodesBuffer,
            normalizedToVoxelCoord(setup.rayOriginNormalized),
            MAX_DEPTH,
            0.0,
            setup.c0,
            setup.c1,
            setup.mirrorMask,
            currentVoxelCoord,
            currentTMin,
            currentNodeIndex,
            currentChildIndex
        );

        for (uint i = 0; i < WORLD_VOXEL_COUNT; i++) {
            if (depth == MAX_DEPTH) {
                SVOLeaf leaf = leavesBuffer[currentNodeIndex];
                uint childIndex = currentChildIndex ^ setup.mirrorMaskLeaf;
                if (leaf.getVoxel(childIndex)) {
                    return true;
                }
            } else {
                if (nodeIsFilled(nodesBuffer, currentNodeIndex, currentChildIndex ^ setup.mirrorMask)) {
                    return true;
                }
            }

            float3 t = setup.c0 * (currentTMin + childExtentForDepth(depth)) + setup.c1;
            float tcMax = min(t.x, min(t.y, t.z));

            uint next;
            if (tcMax == t.x) {
                next = 0u;
            } else if (tcMax == t.y) {
                next = 1u;
            } else {
                next = 2u;
            }

            uint nextFixedDepth = depth;
            if (depth == MAX_DEPTH) {
                if (((currentChildIndex >> (next << 1)) & 0x3u) == 0x3u) {
                    int ancestorDepth = findAncestorDepthForAxis(currentVoxelCoord, next, depth);
                    if (ancestorDepth <= -1) {
                        return false;
                    }
                    nextFixedDepth = uint(ancestorDepth);
                }
            } else if (((currentChildIndex >> next) & 1u) != 0u) {
                int ancestorDepth = findAncestorDepthForAxis(currentVoxelCoord, next, depth);
                if (ancestorDepth <= -1) {
                    return false;
                }
                nextFixedDepth = uint(ancestorDepth);
            }

            uint3 nextFixedVoxelCoord = currentVoxelCoord;
            uint cellExtent = cellVoxelExtentForDepth(depth);
            if (next == 0u) {
                nextFixedVoxelCoord.x += cellExtent;
            } else if (next == 1u) {
                nextFixedVoxelCoord.y += cellExtent;
            } else {
                nextFixedVoxelCoord.z += cellExtent;
            }

            nextFixedVoxelCoord = alignVoxelCoordToDepth(nextFixedVoxelCoord, nextFixedDepth);

            depth = RootRetraceRaycastImpl.retraceFromRoot(
                nodesBuffer,
                nextFixedVoxelCoord,
                nextFixedDepth,
                tcMax,
                setup.c0,
                setup.c1,
                setup.mirrorMask,
                currentVoxelCoord,
                currentTMin,
                currentNodeIndex,
                currentChildIndex
            );
        }

        return false;
    }
}
