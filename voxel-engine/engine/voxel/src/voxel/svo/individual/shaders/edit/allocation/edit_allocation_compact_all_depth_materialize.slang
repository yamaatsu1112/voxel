module edit_allocation_compact_all_depth_materialize;

import svo;
import edit_allocation_compact_all_depth;

public void initializeCompactAllDepthMaterialized(
    RWStructuredBuffer<SVOIndividualNode> nodesBuffer,
    RWStructuredBuffer<SVOLeaf> leavesBuffer,
    RWStructuredBuffer<uint> freeNodeIndicesBuffer,
    RWStructuredBuffer<uint> freeLeafIndicesBuffer,
    RWStructuredBuffer<NodeRequest> stateBuffer,
    RWStructuredBuffer<uint> nodeOffsets,
    RWStructuredBuffer<uint> leafOffsets,
    const uint baseNodeCount,
    const uint baseLeafCount,
    const uint baseFreeNodeCount,
    const uint baseFreeLeafCount,
    const uint threadIndex
) {
    NodeRequest state = stateBuffer[threadIndex];
    uint startDepth = missingStartDepth(state.childIndex);
    if (startDepth == 0u) {
        return;
    }

    bool filled = missingInheritedFilled(state.childIndex);
    uint initializedChildData = filled ? FILLED_BIT : 0u;
    uint base = nodeBase(nodeOffsets, threadIndex);
    for (uint depth = startDepth; depth < MAX_DEPTH; depth++) {
        uint nodeRank = base + (depth - startDepth);
        uint newNodeIndex = materializedNodeIndex(
            freeNodeIndicesBuffer, baseNodeCount, baseFreeNodeCount, nodeRank);
        if (newNodeIndex >= MAX_NODE_COUNT) {
            continue;
        }

        for (uint i = 0u; i < 8u; i++) {
            nodesBuffer[newNodeIndex].childData[i] = initializedChildData;
        }
    }

    uint leafRank = leafBase(leafOffsets, threadIndex);
    uint newLeafIndex = materializedLeafIndex(
        freeLeafIndicesBuffer, baseLeafCount, baseFreeLeafCount, leafRank);
    if (newLeafIndex >= MAX_LEAF_COUNT) {
        return;
    }

    if (filled) {
        leavesBuffer[newLeafIndex].fill();
    } else {
        leavesBuffer[newLeafIndex].clear();
    }
}
