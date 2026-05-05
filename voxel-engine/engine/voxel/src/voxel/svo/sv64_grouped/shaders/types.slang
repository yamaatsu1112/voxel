// SVO Grouped Types - SV64 grouped SVO node definitions.

implementing svo;

public static const uint ROOT_NODE_INDEX = 0u;
public static const uint GROUP_SIZE = 64u;
public static const uint MAX_NODE_COUNT = 65536u;
public static const uint MAX_LEAF_COUNT = 1048576u;
public static const uint LEAF_VOXEL_COUNT_EXP = 2u;
public static const uint LEAF_VOXEL_COUNT = 1u << LEAF_VOXEL_COUNT_EXP;
public static const uint BRANCH_FACTOR_EXP = 2u;
public static const uint MAX_DEPTH =
    (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / BRANCH_FACTOR_EXP;
public static const float LEAF_VOXEL_COUNT_INV = 1.0 / float(LEAF_VOXEL_COUNT);
public static const uint WORLD_VOXEL_COUNT = 1u << WORLD_VOXEL_COUNT_EXP;
public static const uint CHILD_INDEX_OFFSET = 0u;
public static const uint CHILD_INDEX_MASK = 0x3FFFFFFFu;
public static const uint GROUPED_FILLED_MASK = 0xFFFFFFFFu;

public struct NodeRequest {
    public uint nodeIndex;
    public uint maskLow;
    public uint maskHigh;
};

public float nodeSizeForDepth(uint d) {
    return pow(2.0, -float(BRANCH_FACTOR_EXP * d));
}

public float childExtentForDepth(uint depth) {
    if (depth < MAX_DEPTH) {
        return nodeSizeForDepth(depth + 1);
    }

    return nodeSizeForDepth(MAX_DEPTH) * LEAF_VOXEL_COUNT_INV;
}

// ============================================================================
// SVOGroupedNode
// ============================================================================

public struct SVOGroupedNode {
    public uint childMask[2];
    public uint filledMask[2];
    public uint headPointer;

    [mutating]
    public void setChildMask(uint childIndex, bool value) {
        uint word = childIndex < 32u ? 0u : 1u;
        uint bit = 1u << (childIndex & 31u);
        if (value) {
            this.childMask[word] |= bit;
        } else {
            this.childMask[word] &= ~bit;
        }
    }

    [mutating]
    public void setFilled(uint childIndex, bool value) {
        uint word = childIndex < 32u ? 0u : 1u;
        uint bit = 1u << (childIndex & 31u);
        if (value) {
            this.filledMask[word] |= bit;
        } else {
            this.filledMask[word] &= ~bit;
        }
    }

    public bool hasChild(uint childIndex) {
        uint word = childIndex < 32u ? 0u : 1u;
        uint bit = 1u << (childIndex & 31u);
        return (this.childMask[word] & bit) != 0;
    }

    public uint getChildNodeIndex(uint childIndex) {
        return this.headPointer + childIndex;
    }

    public bool isFilled(uint childIndex) {
        uint word = childIndex < 32u ? 0u : 1u;
        uint bit = 1u << (childIndex & 31u);
        return (this.filledMask[word] & bit) != 0;
    }

    [mutating]
    public void setUniform(bool value) {
        this.headPointer = 0u;
        this.childMask[0] = 0u;
        this.childMask[1] = 0u;
        this.filledMask[0] = value ? GROUPED_FILLED_MASK : 0u;
        this.filledMask[1] = value ? GROUPED_FILLED_MASK : 0u;
    }

    public bool isAllEmpty() {
        return (this.childMask[0] == 0u && this.childMask[1] == 0u &&
            this.filledMask[0] == 0u && this.filledMask[1] == 0u);
    }

    public bool isAllFull() {
        return (this.childMask[0] == 0u && this.childMask[1] == 0u &&
            this.filledMask[0] == GROUPED_FILLED_MASK &&
            this.filledMask[1] == GROUPED_FILLED_MASK);
    }
};

public uint nodeChildMask(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint word) {
    return nodesBuffer[nodeIndex].childMask[word];
}

public uint nodeChildMask(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint word) {
    return nodesBuffer[nodeIndex].childMask[word];
}

public uint nodeFilledMask(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint word) {
    return nodesBuffer[nodeIndex].filledMask[word];
}

public uint nodeFilledMask(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint word) {
    return nodesBuffer[nodeIndex].filledMask[word];
}

public uint nodeHeadPointer(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodesBuffer[nodeIndex].headPointer;
}

public uint nodeHeadPointer(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodesBuffer[nodeIndex].headPointer;
}

public bool nodeHasChild(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    uint word = childIndex < 32u ? 0u : 1u;
    uint bit = 1u << (childIndex & 31u);
    return (nodeChildMask(nodesBuffer, nodeIndex, word) & bit) != 0;
}

public bool nodeHasChild(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    uint word = childIndex < 32u ? 0u : 1u;
    uint bit = 1u << (childIndex & 31u);
    return (nodeChildMask(nodesBuffer, nodeIndex, word) & bit) != 0;
}

public uint nodeChildIndex(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return nodeHeadPointer(nodesBuffer, nodeIndex) + childIndex;
}

public uint nodeChildIndex(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return nodeHeadPointer(nodesBuffer, nodeIndex) + childIndex;
}

public bool nodeIsFilled(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    uint word = childIndex < 32u ? 0u : 1u;
    uint bit = 1u << (childIndex & 31u);
    return (nodeFilledMask(nodesBuffer, nodeIndex, word) & bit) != 0;
}

public bool nodeIsFilled(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    uint word = childIndex < 32u ? 0u : 1u;
    uint bit = 1u << (childIndex & 31u);
    return (nodeFilledMask(nodesBuffer, nodeIndex, word) & bit) != 0;
}

public bool nodeIsAllEmpty(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodeChildMask(nodesBuffer, nodeIndex, 0u) == 0u &&
        nodeChildMask(nodesBuffer, nodeIndex, 1u) == 0u &&
        nodeFilledMask(nodesBuffer, nodeIndex, 0u) == 0u &&
        nodeFilledMask(nodesBuffer, nodeIndex, 1u) == 0u;
}

public bool nodeIsAllFull(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodeChildMask(nodesBuffer, nodeIndex, 0u) == 0u &&
        nodeChildMask(nodesBuffer, nodeIndex, 1u) == 0u &&
        nodeFilledMask(nodesBuffer, nodeIndex, 0u) == GROUPED_FILLED_MASK &&
        nodeFilledMask(nodesBuffer, nodeIndex, 1u) == GROUPED_FILLED_MASK;
}
