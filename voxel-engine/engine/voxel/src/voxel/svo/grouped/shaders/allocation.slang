// SVO Grouped Allocation - Grouped SVO allocation and free functions
// This module contains grouped variant methods for block allocation and freeing.

implementing svo;

public static const uint INVALID_REQUEST_NODE_INDEX = 0xFFFFFFFFu;

public bool hasNodeRequest(NodeRequest request) {
    return request.nodeIndex != INVALID_REQUEST_NODE_INDEX;
}

public bool sameNodeRequest(NodeRequest a, NodeRequest b) {
    return a.nodeIndex == b.nodeIndex;
}

public void clearNodeRequest(
    RWStructuredBuffer<NodeRequest> requestBuffer,
    uint threadIndex
) {
    NodeRequest request;
    request.nodeIndex = INVALID_REQUEST_NODE_INDEX;
    request.maskLow = 0u;
    request.maskHigh = 0u;
    requestBuffer[threadIndex] = request;
}

public void resetGroupedSVO(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    RWStructuredBuffer<SVOCounters> countersBuffer
) {
    nodesBuffer[ROOT_NODE_INDEX].headPointer = 0u;
    nodesBuffer[ROOT_NODE_INDEX].masks = 0u;
    countersBuffer[0].nodeCount = 1u;
    countersBuffer[0].leafCount = 1u;
    countersBuffer[0].freeNodeCount = 0u;
    countersBuffer[0].freeLeafCount = 0u;
}

public void collectGroupedAllocationRequests(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    StructuredBuffer<LeafEditCommand> editCommands,
    RWStructuredBuffer<NodeRequest> requestBuffer,
    RWStructuredBuffer<uint> requestCountBuffer,
    uint depth,
    uint threadIndex
) {
    LeafEditCommand cmd = editCommands[threadIndex];

    uint leafWorldX = cmd.leaf_x * LEAF_VOXEL_COUNT;
    uint leafWorldY = cmd.leaf_y * LEAF_VOXEL_COUNT;
    uint leafWorldZ = cmd.leaf_z * LEAF_VOXEL_COUNT;

    uint nodeIndex = ROOT_NODE_INDEX;
    for (uint d = 0u; d < depth - 1; d++) {
        uint childIndex = getChildIndex(leafWorldX, leafWorldY, leafWorldZ, d);
        nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, childIndex);
    }

    uint childIndex = getChildIndex(leafWorldX, leafWorldY, leafWorldZ, depth - 1);

    if (!nodeHasChild(nodesBuffer, nodeIndex, childIndex)) {
        if ((nodeMasks(nodesBuffer, nodeIndex) & 0xFFu) == 0u) {
            NodeRequest pair;
            pair.nodeIndex = nodeIndex;
            pair.maskLow = 1u << childIndex;
            pair.maskHigh = 0u;
            requestBuffer[threadIndex] = pair;
            requestCountBuffer[threadIndex] = 1u;
        } else {
            uint childMask = 1u << childIndex;
            uint filledMask = nodeIsFilled(nodesBuffer, nodeIndex, childIndex) ? (1u << (childIndex + 8u)) : 0u;
            uint unused = 0u;
            InterlockedOr(nodesBuffer[nodeIndex].masks, childMask | filledMask, unused);
            clearNodeRequest(requestBuffer, threadIndex);
            requestCountBuffer[threadIndex] = 0u;
        }
    } else {
        clearNodeRequest(requestBuffer, threadIndex);
            requestCountBuffer[threadIndex] = 0u;
    }
}

public void updateGroupedLeafValues<P : ILeafEditPolicy>(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    RWStructuredBuffer<SVOLeaf> leavesBuffer,
    StructuredBuffer<LeafEditCommand> editCommands,
    uint threadIndex
) {
    LeafEditCommand cmd = editCommands[threadIndex];

    uint leafWorldX = cmd.leaf_x * LEAF_VOXEL_COUNT;
    uint leafWorldY = cmd.leaf_y * LEAF_VOXEL_COUNT;
    uint leafWorldZ = cmd.leaf_z * LEAF_VOXEL_COUNT;

    uint nodeIndex = ROOT_NODE_INDEX;
    for (uint d = 0u; d < MAX_DEPTH; d++) {
        uint childIndex = getChildIndex(leafWorldX, leafWorldY, leafWorldZ, d);
        nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, childIndex);
    }

    SVOLeaf leaf = leavesBuffer[nodeIndex];
    leaf.voxelDataLow = P.applyLow(leaf.voxelDataLow, cmd.voxel_data_low);
    leaf.voxelDataHigh = P.applyHigh(leaf.voxelDataHigh, cmd.voxel_data_high);
    leavesBuffer[nodeIndex] = leaf;
}

public void groupedComputeUniqueFlags(
    RWStructuredBuffer<NodeRequest> requestBuffer,
    RWStructuredBuffer<uint> flagBuffer,
    RWStructuredBuffer<uint> prefixSumBuffer,
    uint commandCount,
    uint threadIndex
) {
    flagBuffer[threadIndex] = 0u;
    prefixSumBuffer[threadIndex] = 0u;
    if (threadIndex >= commandCount) {
        return;
    }

    NodeRequest curr = requestBuffer[threadIndex];
    if (!hasNodeRequest(curr)) {
        return;
    }

    bool isDifferent = true;
    for (uint j = threadIndex; j > 0u; --j) {
        NodeRequest prev = requestBuffer[j - 1u];
        if (!hasNodeRequest(prev)) {
            continue;
        }
        isDifferent = !sameNodeRequest(curr, prev);
        break;
    }

    flagBuffer[threadIndex] = isDifferent ? 1u : 0u;
    prefixSumBuffer[threadIndex] = isDifferent ? 1u : 0u;
}

public void groupedCompactUniquePairs(
    RWStructuredBuffer<NodeRequest> inputBuffer,
    RWStructuredBuffer<NodeRequest> outputBuffer,
    RWStructuredBuffer<uint> flagBuffer,
    RWStructuredBuffer<uint> prefixSumBuffer,
    uint commandCount,
    uint threadIndex
) {
    if (threadIndex >= commandCount) {
        return;
    }

    if (flagBuffer[threadIndex] != 1u) {
        return;
    }

    uint nodeIndex = inputBuffer[threadIndex].nodeIndex;
    uint mergedMaskLow = inputBuffer[threadIndex].maskLow;
    uint mergedMaskHigh = inputBuffer[threadIndex].maskHigh;
    for (uint j = threadIndex + 1u; j < commandCount; ++j) {
        NodeRequest next = inputBuffer[j];
        if (!hasNodeRequest(next)) {
            continue;
        }
        if (next.nodeIndex != nodeIndex) {
            break;
        }

        mergedMaskLow |= next.maskLow;
        mergedMaskHigh |= next.maskHigh;
    }

    uint outputIndex = prefixSumBuffer[threadIndex] - 1u;
    NodeRequest compacted;
    compacted.nodeIndex = nodeIndex;
    compacted.maskLow = mergedMaskLow;
    compacted.maskHigh = mergedMaskHigh;
    outputBuffer[outputIndex] = compacted;
}

public void initializeGroupedNodeBlock(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    uint parentIndex,
    uint blockIndex
) {
    uint parentMasks = nodeMasks(nodesBuffer, parentIndex);

    for (uint childIndex = 0u; childIndex < GROUP_SIZE; childIndex++) {
        bool filled = (parentMasks & (1u << (childIndex + 8u))) != 0;
        SVOGroupedNode node;
        node.headPointer = 0u;
        node.masks = filled ? GROUPED_FILLED_MASK : 0u;
        nodesBuffer[blockIndex + childIndex] = node;
    }
}

public void initializeGroupedLeafBlock(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    RWStructuredBuffer<SVOLeaf> leavesBuffer,
    uint parentIndex,
    uint blockIndex
) {
    uint parentMasks = nodeMasks(nodesBuffer, parentIndex);

    for (uint childIndex = 0u; childIndex < GROUP_SIZE; childIndex++) {
        bool filled = (parentMasks & (1u << (childIndex + 8u))) != 0;
        if (filled) {
            leavesBuffer[blockIndex + childIndex].fill();
        } else {
            leavesBuffer[blockIndex + childIndex].clear();
        }
    }
}

public void allocateGroupedNodesBatch(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    RWStructuredBuffer<SVOLeaf> leavesBuffer,
    RWStructuredBuffer<uint> freeNodeIndicesBuffer,
    RWStructuredBuffer<uint> freeLeafIndicesBuffer,
    RWStructuredBuffer<NodeRequest> uniquePairs,
    const uint depth,
    const uint baseNodeCount,
    const uint baseLeafCount,
    const uint baseFreeNodeCount,
    const uint baseFreeLeafCount,
    const uint threadIndex
) {
    NodeRequest pair = uniquePairs[threadIndex];
    uint mergedChildMask = pair.maskLow & 0xFFu;
    uint parentMasks = nodeMasks(nodesBuffer, pair.nodeIndex);
    uint inheritedFilledMask = 0u;
    for (uint i = 0u; i < GROUP_SIZE; i++) {
        uint childBit = 1u << i;
        uint filledBit = 1u << (i + 8u);
        if ((mergedChildMask & childBit) != 0u && (parentMasks & filledBit) != 0u) {
            inheritedFilledMask |= filledBit;
        }
    }

    if (depth < MAX_DEPTH) {
        uint blockIndex = threadIndex < baseFreeNodeCount
            ? freeNodeIndicesBuffer[baseFreeNodeCount - 1u - threadIndex]
            : baseNodeCount + (threadIndex - baseFreeNodeCount) * GROUP_SIZE;
        if (blockIndex + GROUP_SIZE > MAX_NODE_COUNT) {
            return;
        }

        initializeGroupedNodeBlock(nodesBuffer, pair.nodeIndex, blockIndex);
        nodesBuffer[pair.nodeIndex].headPointer = blockIndex;
        nodesBuffer[pair.nodeIndex].masks = (parentMasks & 0xFF00u) | mergedChildMask | inheritedFilledMask;
    } else {
        uint blockIndex = threadIndex < baseFreeLeafCount
            ? freeLeafIndicesBuffer[baseFreeLeafCount - 1u - threadIndex]
            : baseLeafCount + (threadIndex - baseFreeLeafCount) * GROUP_SIZE;
        if (blockIndex + GROUP_SIZE > MAX_LEAF_COUNT) {
            return;
        }

        initializeGroupedLeafBlock(nodesBuffer, leavesBuffer, pair.nodeIndex, blockIndex);
        nodesBuffer[pair.nodeIndex].headPointer = blockIndex;
        nodesBuffer[pair.nodeIndex].masks = (parentMasks & 0xFF00u) | mergedChildMask | inheritedFilledMask;
    }
}

public void collectGroupedFreeRequests(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    RWStructuredBuffer<SVOLeaf> leavesBuffer,
    StructuredBuffer<LeafEditCommand> editCommands,
    RWStructuredBuffer<NodeRequest> requestBuffer,
    RWStructuredBuffer<uint> requestCountBuffer,
    uint depth,
    uint threadIndex
) {
    if (depth == 1u) {
        clearNodeRequest(requestBuffer, threadIndex);
            requestCountBuffer[threadIndex] = 0u;
        return;
    }

    LeafEditCommand cmd = editCommands[threadIndex];
    uint leafWorldX = cmd.leaf_x * LEAF_VOXEL_COUNT;
    uint leafWorldY = cmd.leaf_y * LEAF_VOXEL_COUNT;
    uint leafWorldZ = cmd.leaf_z * LEAF_VOXEL_COUNT;

    uint nodeIndex = ROOT_NODE_INDEX;
    uint parentIndex = ROOT_NODE_INDEX;
    uint parentChildIndex = 0u;

    for (uint d = 0u; d < depth - 1u; d++) {
        uint childIndex = getChildIndex(leafWorldX, leafWorldY, leafWorldZ, d);

        if (!nodeHasChild(nodesBuffer, nodeIndex, childIndex)) {
            clearNodeRequest(requestBuffer, threadIndex);
            requestCountBuffer[threadIndex] = 0u;
            return;
        }

        if (d == depth - 2u) {
            parentIndex = nodeIndex;
            parentChildIndex = childIndex;
        }

        nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, childIndex);
    }

    uint ownerHeadPointer = nodeHeadPointer(nodesBuffer, nodeIndex);
    if (ownerHeadPointer == 0u) {
        clearNodeRequest(requestBuffer, threadIndex);
        requestCountBuffer[threadIndex] = 0u;
        return;
    }

    bool allEmpty = true;
    bool allFull = true;

    if (depth == MAX_DEPTH) {
        for (uint childIndex = 0u; childIndex < GROUP_SIZE; childIndex++) {
            SVOLeaf leaf = leavesBuffer[ownerHeadPointer + childIndex];
            if (!leaf.isEmpty()) {
                allEmpty = false;
            }
            if (!leaf.isFull()) {
                allFull = false;
            }
            if (!allEmpty && !allFull) {
                clearNodeRequest(requestBuffer, threadIndex);
                requestCountBuffer[threadIndex] = 0u;
                return;
            }
        }
    } else {
        allEmpty = nodeIsAllEmpty(nodesBuffer, nodeIndex);
        allFull = nodeIsAllFull(nodesBuffer, nodeIndex);
        if (!allEmpty && !allFull) {
            clearNodeRequest(requestBuffer, threadIndex);
            requestCountBuffer[threadIndex] = 0u;
            return;
        }
    }

    NodeRequest pair;
    pair.nodeIndex = parentIndex;
    pair.maskLow = 1u << parentChildIndex;
    pair.maskHigh = 0u;
    requestBuffer[threadIndex] = pair;
    requestCountBuffer[threadIndex] = 1u;
}

public void freeGroupedLeavesBatch(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    RWStructuredBuffer<SVOLeaf> leavesBuffer,
    RWStructuredBuffer<SVOCounters> countersBuffer,
    RWStructuredBuffer<uint> freeLeafIndicesBuffer,
    RWStructuredBuffer<NodeRequest> uniquePairs,
    const uint threadIndex
) {
    NodeRequest pair = uniquePairs[threadIndex];
    uint mergedChildMask = pair.maskLow & 0xFFu;
    uint freedLeafCount = countbits(mergedChildMask);
    if (freedLeafCount == 0u) {
        return;
    }

    uint freeListIndexBase = 0u;
    InterlockedAdd(countersBuffer[0].freeLeafCount, freedLeafCount, freeListIndexBase);

    uint unused = 0u;
    uint freeListOffset = 0u;
    for (uint childIndex = 0u; childIndex < GROUP_SIZE; childIndex++) {
        uint childBit = 1u << childIndex;
        if ((mergedChildMask & childBit) == 0u) {
            continue;
        }

        uint ownerIndex = nodeChildIndex(nodesBuffer, pair.nodeIndex, childIndex);
        uint ownerHeadPointer = nodeHeadPointer(nodesBuffer, ownerIndex);
        if (ownerHeadPointer == 0u) {
            continue;
        }

        uint freeListIndex = freeListIndexBase + freeListOffset;
        if (freeListIndex < MAX_LEAF_COUNT) {
            freeLeafIndicesBuffer[freeListIndex] = ownerHeadPointer;
        }
        freeListOffset += 1u;

        bool isFull = leavesBuffer[ownerHeadPointer].isFull();
        nodesBuffer[ownerIndex].headPointer = 0u;
        nodesBuffer[ownerIndex].masks = isFull ? GROUPED_FILLED_MASK : 0u;

        InterlockedAnd(nodesBuffer[pair.nodeIndex].masks, ~childBit, unused);
        uint filledBit = 1u << (childIndex + 8u);
        if (isFull) {
            InterlockedOr(nodesBuffer[pair.nodeIndex].masks, filledBit, unused);
        } else {
            InterlockedAnd(nodesBuffer[pair.nodeIndex].masks, ~filledBit, unused);
        }
    }
}

public void freeGroupedNodesBatch(
    RWStructuredBuffer<SVOGroupedNode> nodesBuffer,
    RWStructuredBuffer<SVOCounters> countersBuffer,
    RWStructuredBuffer<uint> freeNodeIndicesBuffer,
    RWStructuredBuffer<NodeRequest> uniquePairs,
    const uint threadIndex
) {
    NodeRequest pair = uniquePairs[threadIndex];
    uint mergedChildMask = pair.maskLow & 0xFFu;
    uint freedNodeCount = countbits(mergedChildMask);
    if (freedNodeCount == 0u) {
        return;
    }

    uint freeListIndexBase = 0u;
    InterlockedAdd(countersBuffer[0].freeNodeCount, freedNodeCount, freeListIndexBase);

    uint unused = 0u;
    uint freeListOffset = 0u;
    for (uint childIndex = 0u; childIndex < GROUP_SIZE; childIndex++) {
        uint childBit = 1u << childIndex;
        if ((mergedChildMask & childBit) == 0u) {
            continue;
        }

        uint ownerIndex = nodeChildIndex(nodesBuffer, pair.nodeIndex, childIndex);
        uint ownerHeadPointer = nodeHeadPointer(nodesBuffer, ownerIndex);
        if (ownerHeadPointer == 0u) {
            continue;
        }

        uint freeListIndex = freeListIndexBase + freeListOffset;
        if (freeListIndex < MAX_NODE_COUNT) {
            freeNodeIndicesBuffer[freeListIndex] = ownerHeadPointer;
        }
        freeListOffset += 1u;

        bool isFull = nodeIsAllFull(nodesBuffer, ownerIndex);
        nodesBuffer[ownerIndex].headPointer = 0u;
        nodesBuffer[ownerIndex].masks = isFull ? GROUPED_FILLED_MASK : 0u;

        InterlockedAnd(nodesBuffer[pair.nodeIndex].masks, ~childBit, unused);
        uint filledBit = 1u << (childIndex + 8u);
        if (isFull) {
            InterlockedOr(nodesBuffer[pair.nodeIndex].masks, filledBit, unused);
        } else {
            InterlockedAnd(nodesBuffer[pair.nodeIndex].masks, ~filledBit, unused);
        }
    }
}
