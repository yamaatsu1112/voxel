// SVO Grouped Types - Grouped SVO node definitions
// This module contains the grouped SVO variant data structures.

implementing svo;

public static const uint ROOT_NODE_INDEX = 0u;
public static const uint GROUP_SIZE = 8u;
public static const uint MAX_NODE_COUNT = 131072u;
public static const uint MAX_LEAF_COUNT = 524288u;
public static const uint LEAF_VOXEL_COUNT_EXP = 2u;
public static const uint LEAF_VOXEL_COUNT = 1u << LEAF_VOXEL_COUNT_EXP;
public static const uint BRANCH_FACTOR_EXP = 1u;
public static const uint MAX_DEPTH =
    (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / BRANCH_FACTOR_EXP;
public static const float LEAF_VOXEL_COUNT_INV = 1.0 / float(LEAF_VOXEL_COUNT);
public static const uint WORLD_VOXEL_COUNT = 1u << WORLD_VOXEL_COUNT_EXP;
public static const uint CHILD_INDEX_OFFSET = 0u;
public static const uint CHILD_MASK_OFFSET = 30u;
public static const uint FILLED_OFFSET = 31u;
public static const uint CHILD_INDEX_MASK = 0x3FFFFFFFu;
public static const uint CHILD_MASK_BIT = 1u << CHILD_MASK_OFFSET;
public static const uint FILLED_BIT = 1u << FILLED_OFFSET;
public static const uint GROUPED_FILLED_MASK = 0xFF00u;

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
    public uint headPointer;
    public uint masks;

    public bool hasChild(uint childIndex) {
        return (this.masks & (1u << childIndex)) != 0;
    }

    public uint childMaskBits() {
        return this.masks & 0xFFu;
    }

    public uint getChildNodeIndex(uint childIndex) {
        return this.headPointer + childIndex;
    }

    public bool isFilled(uint childIndex) {
        return (this.masks & (1u << (childIndex + 8u))) != 0;
    }

    [mutating]
    public void setChildMask(uint childIndex, bool value) {
        uint bit = 1u << childIndex;
        if (value) {
            this.masks |= bit;
        } else {
            this.masks &= ~bit;
        }
    }

    [mutating]
    public void setFilled(uint childIndex, bool value) {
        uint bit = 1u << (childIndex + 8u);
        if (value) {
            this.masks |= bit;
        } else {
            this.masks &= ~bit;
        }
    }

    [mutating]
    public void setUniform(bool value) {
        this.headPointer = 0u;
        this.masks = value ? GROUPED_FILLED_MASK : 0u;
    }

    public bool isAllEmpty() {
        return this.masks == 0u;
    }

    public bool isAllFull() {
        return this.masks == GROUPED_FILLED_MASK;
    }
};

public uint nodeMasks(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodesBuffer[nodeIndex].masks;
}

public uint nodeMasks(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodesBuffer[nodeIndex].masks;
}

public uint nodeHeadPointer(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodesBuffer[nodeIndex].headPointer;
}

public uint nodeHeadPointer(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodesBuffer[nodeIndex].headPointer;
}

public bool nodeHasChild(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return (nodeMasks(nodesBuffer, nodeIndex) & (1u << childIndex)) != 0;
}

public bool nodeHasChild(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return (nodeMasks(nodesBuffer, nodeIndex) & (1u << childIndex)) != 0;
}

public uint nodeChildIndex(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return nodeHeadPointer(nodesBuffer, nodeIndex) + childIndex;
}

public uint nodeChildIndex(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return nodeHeadPointer(nodesBuffer, nodeIndex) + childIndex;
}

public bool nodeIsFilled(StructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return (nodeMasks(nodesBuffer, nodeIndex) & (1u << (childIndex + 8u))) != 0;
}

public bool nodeIsFilled(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return (nodeMasks(nodesBuffer, nodeIndex) & (1u << (childIndex + 8u))) != 0;
}

public bool nodeIsAllEmpty(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodeMasks(nodesBuffer, nodeIndex) == 0u;
}

public bool nodeIsAllFull(RWStructuredBuffer<SVOGroupedNode> nodesBuffer, uint nodeIndex) {
    return nodeMasks(nodesBuffer, nodeIndex) == GROUPED_FILLED_MASK;
}
