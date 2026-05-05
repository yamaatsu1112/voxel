// SVO Individual Allocation - Parallel SVO Edit and Free Functions
// This module contains functions for parallel allocation and freeing of SVO nodes/leaves.

implementing svo;

public static const uint INVALID_REQUEST_NODE_INDEX = 0xFFFFFFFFu;

public bool hasNodeRequest(NodeRequest request) {
    return request.nodeIndex != INVALID_REQUEST_NODE_INDEX;
}

public bool sameNodeRequest(NodeRequest a, NodeRequest b) {
    return a.nodeIndex == b.nodeIndex && a.childIndex == b.childIndex;
}

public void clearNodeRequest(
    RWStructuredBuffer<NodeRequest> requestBuffer,
    uint threadIndex
) {
    NodeRequest request;
    request.nodeIndex = INVALID_REQUEST_NODE_INDEX;
    request.childIndex = 0u;
    requestBuffer[threadIndex] = request;
}

public void prefixSumSharedLocal(
    RWStructuredBuffer<uint> prefixSumBuffer,
    RWStructuredBuffer<uint> requestCountBuffer,
    uint paddedSize,
    uint localIndex,
    uint groupIndex,
    uint threadIndex,
    inout uint sharedData[64]
) {
    sharedData[localIndex] = (threadIndex < paddedSize) ? prefixSumBuffer[threadIndex] : 0u;
    GroupMemoryBarrierWithGroupSync();

    inclusivePrefixSumShared<64, 1>(sharedData, localIndex, 64u);

    if (threadIndex < paddedSize) {
        prefixSumBuffer[threadIndex] = sharedData[localIndex];
    }

    if (localIndex == 63u) {
        requestCountBuffer[1u + groupIndex] = sharedData[63u];
    }
}

public void prefixSumSharedBlockScan(
    RWStructuredBuffer<uint> requestCountBuffer,
    uint numWG,
    uint localIndex,
    inout uint sharedData[64]
) {
    sharedData[localIndex] = (localIndex < numWG) ? requestCountBuffer[1u + localIndex] : 0u;
    GroupMemoryBarrierWithGroupSync();

    inclusivePrefixSumShared<64, 1>(sharedData, localIndex, 64u);

    if (localIndex < numWG) {
        requestCountBuffer[1u + localIndex] = sharedData[localIndex];
    }
}

public void prefixSumSharedPropagate(
    RWStructuredBuffer<uint> prefixSumBuffer,
    RWStructuredBuffer<uint> requestCountBuffer,
    uint paddedSize,
    uint localIndex,
    uint groupIndex,
    uint threadIndex
) {
    if (groupIndex > 0u && threadIndex < paddedSize) {
        prefixSumBuffer[threadIndex] += requestCountBuffer[groupIndex];
    }
}

public void resetIndividualSVO(
    RWStructuredBuffer<SV64IndividualNode> nodesBuffer,
    RWStructuredBuffer<SVOCounters> countersBuffer
) {
    for (uint i = 0u; i < 64u; i++) {
        nodesBuffer[ROOT_NODE_INDEX].childData[i] = 0u;
    }
    countersBuffer[0].nodeCount = 1u;
    countersBuffer[0].leafCount = 0u;
    countersBuffer[0].freeNodeCount = 0u;
    countersBuffer[0].freeLeafCount = 0u;
}

public void collectIndividualAllocationRequests(
    RWStructuredBuffer<SV64IndividualNode> nodesBuffer,
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
        NodeRequest pair;
        pair.nodeIndex = nodeIndex;
        pair.childIndex = childIndex;
        requestBuffer[threadIndex] = pair;
        requestCountBuffer[threadIndex] = 1u;
    } else {
        clearNodeRequest(requestBuffer, threadIndex);
            requestCountBuffer[threadIndex] = 0u;
    }
}

public void updateIndividualLeafValues<P : ILeafEditPolicy>(
    RWStructuredBuffer<SV64IndividualNode> nodesBuffer,
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

public void computeUniqueFlags(
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

public void compactUniquePairs(
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

    if (flagBuffer[threadIndex] == 1u) {
        uint outputIndex = prefixSumBuffer[threadIndex] - 1;
        outputBuffer[outputIndex] = inputBuffer[threadIndex];
    }
}

public void allocateIndividualNodesBatch(
    RWStructuredBuffer<SV64IndividualNode> nodesBuffer,
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

    if (depth < MAX_DEPTH) {
        uint newNodeIndex = (threadIndex < baseFreeNodeCount)
            ? freeNodeIndicesBuffer[baseFreeNodeCount - 1u - threadIndex]
            : baseNodeCount + (threadIndex - baseFreeNodeCount);
        if (newNodeIndex >= MAX_NODE_COUNT) {
            return;
        }

        bool parentFilled = nodeIsFilled(nodesBuffer, pair.nodeIndex, pair.childIndex);

        uint initializedChildData = parentFilled ? FILLED_BIT : 0u;
        for (uint i = 0u; i < 64u; i++) {
            nodesBuffer[newNodeIndex].childData[i] = initializedChildData;
        }

        uint newChildData = (parentFilled ? FILLED_BIT : 0u) | CHILD_MASK_BIT | (newNodeIndex & CHILD_INDEX_MASK);
        nodesBuffer[pair.nodeIndex].childData[pair.childIndex] = newChildData;
    } else {
        uint newLeafIndex = threadIndex < baseFreeLeafCount
            ? freeLeafIndicesBuffer[baseFreeLeafCount - 1u - threadIndex]
            : baseLeafCount + (threadIndex - baseFreeLeafCount);
        if (newLeafIndex >= MAX_LEAF_COUNT) {
            return;
        }

        bool parentFilled = nodeIsFilled(nodesBuffer, pair.nodeIndex, pair.childIndex);

        uint newChildData = (parentFilled ? FILLED_BIT : 0u) | CHILD_MASK_BIT | (newLeafIndex & CHILD_INDEX_MASK);
        nodesBuffer[pair.nodeIndex].childData[pair.childIndex] = newChildData;

        if (parentFilled) {
            leavesBuffer[newLeafIndex].fill();
        } else {
            leavesBuffer[newLeafIndex].clear();
        }
    }
}

public void collectIndividualFreeRequests(
    RWStructuredBuffer<SV64IndividualNode> nodesBuffer,
    RWStructuredBuffer<SVOLeaf> leavesBuffer,
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

    if (depth == MAX_DEPTH) {
        uint nodeIndex = ROOT_NODE_INDEX;
        uint parentIndex = ROOT_NODE_INDEX;
        uint parentChildIndex = 0u;

        for (uint d = 0u; d < MAX_DEPTH; d++) {
            uint childIndex = getChildIndex(leafWorldX, leafWorldY, leafWorldZ, d);

            if (!nodeHasChild(nodesBuffer, nodeIndex, childIndex)) {
                clearNodeRequest(requestBuffer, threadIndex);
                requestCountBuffer[threadIndex] = 0u;
                return;
            }

            if (d == MAX_DEPTH - 1u) {
                parentIndex = nodeIndex;
                parentChildIndex = childIndex;
            }

            nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, childIndex);
        }

        SVOLeaf leaf = leavesBuffer[nodeIndex];

        if (leaf.isEmpty() || leaf.isFull()) {
            NodeRequest pair;
            pair.nodeIndex = parentIndex;
            pair.childIndex = parentChildIndex;
            requestBuffer[threadIndex] = pair;
            requestCountBuffer[threadIndex] = 1u;
        } else {
            clearNodeRequest(requestBuffer, threadIndex);
            requestCountBuffer[threadIndex] = 0u;
        }
    } else {
        uint nodeIndex = ROOT_NODE_INDEX;
        uint parentIndex = ROOT_NODE_INDEX;
        uint parentChildIndex = 0u;

        for (uint d = 0u; d < depth; d++) {
            uint childIndex = getChildIndex(leafWorldX, leafWorldY, leafWorldZ, d);

            if (!nodeHasChild(nodesBuffer, nodeIndex, childIndex)) {
                clearNodeRequest(requestBuffer, threadIndex);
                requestCountBuffer[threadIndex] = 0u;
                return;
            }

            if (d == depth - 1u) {
                parentIndex = nodeIndex;
                parentChildIndex = childIndex;
            }

            nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, childIndex);
        }

        bool allEmpty = true;
        bool allFull = true;

        for (uint i = 0u; i < 64u; i++) {
            if (nodeHasChild(nodesBuffer, nodeIndex, i)) {
                allEmpty = false;
                allFull = false;
                break;
            }
            if (nodeIsFilled(nodesBuffer, nodeIndex, i)) {
                allEmpty = false;
            } else {
                allFull = false;
            }
        }

        if (allEmpty || allFull) {
            NodeRequest pair;
            pair.nodeIndex = parentIndex;
            pair.childIndex = parentChildIndex;
            requestBuffer[threadIndex] = pair;
            requestCountBuffer[threadIndex] = 1u;
        } else {
            clearNodeRequest(requestBuffer, threadIndex);
            requestCountBuffer[threadIndex] = 0u;
        }
    }
}

public void freeIndividualLeavesBatch(
    RWStructuredBuffer<SV64IndividualNode> nodesBuffer,
    RWStructuredBuffer<SVOLeaf> leavesBuffer,
    RWStructuredBuffer<uint> freeLeafIndicesBuffer,
    RWStructuredBuffer<NodeRequest> uniquePairs,
    const uint baseFreeLeafCount,
    const uint threadIndex
) {
    NodeRequest pair = uniquePairs[threadIndex];
    uint parentIndex = pair.nodeIndex;
    uint parentChildIndex = pair.childIndex;

    uint leafIndex = nodeChildIndex(nodesBuffer, parentIndex, parentChildIndex);

    uint freeListIndex = baseFreeLeafCount + threadIndex;
    if (freeListIndex >= MAX_LEAF_COUNT) {
        return;
    }

    freeLeafIndicesBuffer[freeListIndex] = leafIndex;

    bool isFull = leavesBuffer[leafIndex].isFull();
    uint newChildData = isFull ? FILLED_BIT : 0u;
    nodesBuffer[parentIndex].childData[parentChildIndex] = newChildData;
}

public void freeIndividualNodesBatch(
    RWStructuredBuffer<SV64IndividualNode> nodesBuffer,
    RWStructuredBuffer<uint> freeNodeIndicesBuffer,
    RWStructuredBuffer<NodeRequest> uniquePairs,
    const uint baseFreeNodeCount,
    const uint threadIndex
) {
    NodeRequest pair = uniquePairs[threadIndex];
    uint parentIndex = pair.nodeIndex;
    uint parentChildIndex = pair.childIndex;

    uint nodeIndex = nodeChildIndex(nodesBuffer, parentIndex, parentChildIndex);

    uint freeListIndex = baseFreeNodeCount + threadIndex;
    if (freeListIndex >= MAX_NODE_COUNT) {
        return;
    }

    freeNodeIndicesBuffer[freeListIndex] = nodeIndex;

    bool isFull = true;
    for (uint i = 0u; i < 64u; i++) {
        if (nodeHasChild(nodesBuffer, nodeIndex, i) || !nodeIsFilled(nodesBuffer, nodeIndex, i)) {
            isFull = false;
            break;
        }
    }
    uint newChildData = isFull ? FILLED_BIT : 0u;
    nodesBuffer[parentIndex].childData[parentChildIndex] = newChildData;
}
