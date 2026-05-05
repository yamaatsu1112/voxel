module edit_allocation_compact_all_depth_link;

import svo;
import edit_allocation_compact_all_depth;

public uint lowerBoundLeafKey(StructuredBuffer<LeafEditCommand> editCommands, uint count, uint key) {
    uint first = 0u;
    uint remaining = count;
    while (remaining != 0u) {
        uint step = remaining >> 1u;
        uint mid = first + step;
        if (leafKey(editCommands[mid]) < key) {
            first = mid + 1u;
            remaining -= step + 1u;
        } else {
            remaining = step;
        }
    }
    return first;
}

public uint prefixForLeafKey(uint key, uint depth) {
    uint shift = (MAX_DEPTH - depth) * 3u;
    return shift >= 32u ? 0u : key >> shift;
}

public uint emittedNodeRank(
    RWStructuredBuffer<NodeRequest> stateBuffer,
    RWStructuredBuffer<uint> nodeOffsets,
    uint owner,
    uint depth
) {
    uint startDepth = missingStartDepth(stateBuffer[owner].childIndex);
    return nodeBase(nodeOffsets, owner) + (depth - startDepth);
}

public uint compactAllDepthParentIndex(
    RWStructuredBuffer<uint> freeNodeIndicesBuffer,
    StructuredBuffer<LeafEditCommand> editCommands,
    RWStructuredBuffer<NodeRequest> stateBuffer,
    RWStructuredBuffer<uint> nodeOffsets,
    uint commandCount,
    uint baseNodeCount,
    uint baseFreeNodeCount,
    uint owner,
    uint depth,
    uint startDepth,
    uint base
) {
    uint firstMissingDepth = missingFirstDepth(stateBuffer[owner].childIndex);
    if (depth > startDepth) {
        return materializedNodeIndex(
            freeNodeIndicesBuffer, baseNodeCount, baseFreeNodeCount,
            base + (depth - startDepth - 1u));
    }
    if (startDepth == firstMissingDepth) {
        return stateBuffer[owner].nodeIndex;
    }

    uint key = leafKey(editCommands[owner]);
    uint parentDepth = depth - 1u;
    uint parentPrefix = prefixForLeafKey(key, parentDepth);
    uint shift = (MAX_DEPTH - parentDepth) * 3u;
    uint firstKey = parentPrefix << shift;
    uint parentOwner = lowerBoundLeafKey(editCommands, commandCount, firstKey);
    if (parentOwner >= commandCount) {
        return MAX_NODE_COUNT;
    }

    return materializedNodeIndex(
        freeNodeIndicesBuffer, baseNodeCount, baseFreeNodeCount,
        emittedNodeRank(stateBuffer, nodeOffsets, parentOwner, parentDepth));
}

public void linkCompactAllDepthMaterialized(
    RWStructuredBuffer<SVOIndividualNode> nodesBuffer,
    RWStructuredBuffer<uint> freeNodeIndicesBuffer,
    RWStructuredBuffer<uint> freeLeafIndicesBuffer,
    StructuredBuffer<LeafEditCommand> editCommands,
    RWStructuredBuffer<NodeRequest> stateBuffer,
    RWStructuredBuffer<uint> nodeOffsets,
    RWStructuredBuffer<uint> leafOffsets,
    uint commandCount,
    const uint baseNodeCount,
    const uint baseLeafCount,
    const uint baseFreeNodeCount,
    const uint baseFreeLeafCount,
    const uint threadIndex
) {
    NodeRequest state = stateBuffer[threadIndex];
    uint startDepth = missingStartDepth(state.childIndex);
    if (startDepth == 0u || threadIndex >= commandCount) {
        return;
    }

    uint key = leafKey(editCommands[threadIndex]);
    bool filled = missingInheritedFilled(state.childIndex);
    uint newChildDataBase = (filled ? FILLED_BIT : 0u) | CHILD_MASK_BIT;
    uint base = nodeBase(nodeOffsets, threadIndex);

    for (uint depth = startDepth; depth <= MAX_DEPTH; depth++) {
        uint childStorageIndex = 0u;
        if (depth < MAX_DEPTH) {
            childStorageIndex = materializedNodeIndex(
                freeNodeIndicesBuffer, baseNodeCount, baseFreeNodeCount,
                base + (depth - startDepth));
            if (childStorageIndex >= MAX_NODE_COUNT) {
                continue;
            }
        } else {
            childStorageIndex = materializedLeafIndex(
                freeLeafIndicesBuffer, baseLeafCount, baseFreeLeafCount,
                leafBase(leafOffsets, threadIndex));
            if (childStorageIndex >= MAX_LEAF_COUNT) {
                continue;
            }
        }

        uint parentIndex = compactAllDepthParentIndex(
            freeNodeIndicesBuffer,
            editCommands,
            stateBuffer,
            nodeOffsets,
            commandCount,
            baseNodeCount,
            baseFreeNodeCount,
            threadIndex,
            depth,
            startDepth,
            base);
        if (parentIndex >= MAX_NODE_COUNT) {
            continue;
        }

        uint childIndex = childIndexForLeafKey(key, depth - 1u);
        nodesBuffer[parentIndex].childData[childIndex] =
            newChildDataBase | (childStorageIndex & CHILD_INDEX_MASK);
    }
}
