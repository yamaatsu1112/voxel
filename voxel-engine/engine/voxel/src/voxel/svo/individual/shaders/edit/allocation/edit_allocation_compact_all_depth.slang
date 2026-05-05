module edit_allocation_compact_all_depth;

import svo;
import edit_allocation_common;

public static const uint MISSING_FIRST_DEPTH_MASK = 0xFFu;
public static const uint MISSING_FILLED_BIT = 1u << 8u;
public static const uint MISSING_START_DEPTH_SHIFT = 16u;

public uint leafKey(LeafEditCommand cmd) {
    uint key = 0u;
    for (uint bit = 0u; bit < WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP; bit++) {
        key |= ((cmd.leaf_x >> bit) & 1u) << (bit * 3u);
        key |= ((cmd.leaf_y >> bit) & 1u) << (bit * 3u + 1u);
        key |= ((cmd.leaf_z >> bit) & 1u) << (bit * 3u + 2u);
    }
    return key;
}

public uint childIndexForLeafKey(uint key, uint depth) {
    uint shift = (MAX_DEPTH - depth - 1u) * 3u;
    return (key >> shift) & 7u;
}

public uint lcpDepth(uint lhs, uint rhs) {
    uint depth = 0u;
    for (; depth < MAX_DEPTH; depth++) {
        if (childIndexForLeafKey(lhs, depth) != childIndexForLeafKey(rhs, depth)) {
            break;
        }
    }
    return depth;
}

public uint nodeEmitCount(uint startDepth) {
    return startDepth != 0u && startDepth < MAX_DEPTH ? MAX_DEPTH - startDepth : 0u;
}

public uint leafEmitCount(uint startDepth) {
    return startDepth != 0u ? 1u : 0u;
}

public uint packMissingState(uint firstMissingDepth, bool filled, uint startDepth) {
    return firstMissingDepth | (filled ? MISSING_FILLED_BIT : 0u) | (startDepth << MISSING_START_DEPTH_SHIFT);
}

public uint missingFirstDepth(uint packedState) {
    return packedState & MISSING_FIRST_DEPTH_MASK;
}

public bool missingInheritedFilled(uint packedState) {
    return (packedState & MISSING_FILLED_BIT) != 0u;
}

public uint missingStartDepth(uint packedState) {
    return packedState >> MISSING_START_DEPTH_SHIFT;
}

public uint exclusiveOffset(RWStructuredBuffer<uint> offsets, uint index) {
    return index == 0u ? 0u : offsets[index - 1u];
}

public uint nodeBase(RWStructuredBuffer<uint> nodeOffsets, uint index) {
    return exclusiveOffset(nodeOffsets, index);
}

public uint leafBase(RWStructuredBuffer<uint> leafOffsets, uint index) {
    return exclusiveOffset(leafOffsets, index);
}

public uint materializedNodeIndex(
    RWStructuredBuffer<uint> freeNodeIndicesBuffer,
    uint baseNodeCount,
    uint baseFreeNodeCount,
    uint nodeRank
) {
    return nodeRank < baseFreeNodeCount
        ? freeNodeIndicesBuffer[baseFreeNodeCount - 1u - nodeRank]
        : baseNodeCount + (nodeRank - baseFreeNodeCount);
}

public uint materializedLeafIndex(
    RWStructuredBuffer<uint> freeLeafIndicesBuffer,
    uint baseLeafCount,
    uint baseFreeLeafCount,
    uint leafRank
) {
    return leafRank < baseFreeLeafCount
        ? freeLeafIndicesBuffer[baseFreeLeafCount - 1u - leafRank]
        : baseLeafCount + (leafRank - baseFreeLeafCount);
}

public void collectCompactAllDepthCounts(
    RWStructuredBuffer<SVOIndividualNode> nodesBuffer,
    StructuredBuffer<LeafEditCommand> editCommands,
    RWStructuredBuffer<NodeRequest> stateBuffer,
    RWStructuredBuffer<uint> nodeOffsets,
    RWStructuredBuffer<uint> leafOffsets,
    uint commandCount,
    uint threadIndex
) {
    if (threadIndex >= commandCount) {
        clearNodeRequest(stateBuffer, threadIndex);
        nodeOffsets[threadIndex] = 0u;
        leafOffsets[threadIndex] = 0u;
        return;
    }

    LeafEditCommand cmd = editCommands[threadIndex];
    uint key = leafKey(cmd);
    uint nodeIndex = ROOT_NODE_INDEX;
    uint firstMissingDepth = 0u;
    uint existingParentIndex = ROOT_NODE_INDEX;
    bool filled = false;

    for (uint depth = 1u; depth <= MAX_DEPTH; depth++) {
        uint childIndex = childIndexForLeafKey(key, depth - 1u);
        if (!nodeHasChild(nodesBuffer, nodeIndex, childIndex)) {
            firstMissingDepth = depth;
            existingParentIndex = nodeIndex;
            filled = nodeIsFilled(nodesBuffer, nodeIndex, childIndex);
            break;
        }
        nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, childIndex);
    }

    uint startDepth = 0u;
    if (firstMissingDepth != 0u) {
        uint lcp = threadIndex == 0u ? 0u : lcpDepth(key, leafKey(editCommands[threadIndex - 1u]));
        uint uniqueDepth = lcp + 1u;
        startDepth = max(firstMissingDepth, uniqueDepth);
    }

    NodeRequest state;
    state.nodeIndex = existingParentIndex;
    state.childIndex = packMissingState(firstMissingDepth, filled, startDepth);
    stateBuffer[threadIndex] = state;
    nodeOffsets[threadIndex] = nodeEmitCount(startDepth);
    leafOffsets[threadIndex] = leafEmitCount(startDepth);
}
