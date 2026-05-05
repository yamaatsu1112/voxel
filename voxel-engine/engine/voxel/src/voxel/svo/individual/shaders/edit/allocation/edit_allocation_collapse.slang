module edit_allocation_collapse;

import svo;
import edit_allocation_common;

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
public void collectIndividualFreeRequests(
    RWStructuredBuffer<SVOIndividualNode> nodesBuffer,
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

        for (uint i = 0u; i < 8u; i++) {
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
    RWStructuredBuffer<SVOIndividualNode> nodesBuffer,
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
    RWStructuredBuffer<SVOIndividualNode> nodesBuffer,
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
    for (uint i = 0u; i < 8u; i++) {
        if (nodeHasChild(nodesBuffer, nodeIndex, i) || !nodeIsFilled(nodesBuffer, nodeIndex, i)) {
            isFull = false;
            break;
        }
    }
    uint newChildData = isFull ? FILLED_BIT : 0u;
    nodesBuffer[parentIndex].childData[parentChildIndex] = newChildData;
}
