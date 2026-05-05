// SVO Individual Types - Individual SVO node definitions
// This module contains the individual SVO variant data structures.

implementing svo;

public static const uint ROOT_NODE_INDEX = 0u;
public static const uint MAX_NODE_COUNT = 65536u;
public static const uint MAX_LEAF_COUNT = 262144u;
public static const uint LEAF_VOXEL_COUNT_EXP = 2u;
public static const uint LEAF_VOXEL_COUNT = 1u << LEAF_VOXEL_COUNT_EXP;
public static const uint BRANCH_FACTOR_EXP = 2u;
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

public struct NodeRequest {
    public uint nodeIndex;
    public uint childIndex;
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
// SV64IndividualNode
// ============================================================================

public struct SV64IndividualNode {
    public uint childData[64];  // Each uint stores filled, child_mask, and child_index

    public uint getChildNodeIndex(uint childIndex) {
        return this.childData[childIndex] & CHILD_INDEX_MASK;
    }

    [mutating]
    public void setChildMask(uint childIndex, bool value) {
        if (value) {
            this.childData[childIndex] |= CHILD_MASK_BIT;
        } else {
            this.childData[childIndex] &= ~CHILD_MASK_BIT;
        }
    }

    [mutating]
    public void setFilled(uint childIndex, bool value) {
        if (value) {
            this.childData[childIndex] |= FILLED_BIT;
        } else {
            this.childData[childIndex] &= ~FILLED_BIT;
        }
    }

    [mutating]
    public void setChildIndex(uint childIndex, uint index) {
        this.childData[childIndex] = (this.childData[childIndex] & ~CHILD_INDEX_MASK) |
                                     (index & CHILD_INDEX_MASK);
    }

    // Check if a specific child exists
    public bool hasChild(uint childIndex) {
        return (this.childData[childIndex] & CHILD_MASK_BIT) != 0;
    }

    // Check if a specific child is filled (solid)
    public bool isFilled(uint childIndex) {
        return (this.childData[childIndex] & FILLED_BIT) != 0;
    }
};

public uint nodeChildData(StructuredBuffer<SV64IndividualNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return nodesBuffer[nodeIndex].childData[childIndex];
}

public uint nodeChildData(RWStructuredBuffer<SV64IndividualNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return nodesBuffer[nodeIndex].childData[childIndex];
}

public uint nodeChildIndex(StructuredBuffer<SV64IndividualNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return nodeChildData(nodesBuffer, nodeIndex, childIndex) & CHILD_INDEX_MASK;
}

public uint nodeChildIndex(RWStructuredBuffer<SV64IndividualNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return nodeChildData(nodesBuffer, nodeIndex, childIndex) & CHILD_INDEX_MASK;
}

public bool nodeHasChild(StructuredBuffer<SV64IndividualNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return (nodeChildData(nodesBuffer, nodeIndex, childIndex) & CHILD_MASK_BIT) != 0;
}

public bool nodeHasChild(RWStructuredBuffer<SV64IndividualNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return (nodeChildData(nodesBuffer, nodeIndex, childIndex) & CHILD_MASK_BIT) != 0;
}

public bool nodeIsFilled(StructuredBuffer<SV64IndividualNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return (nodeChildData(nodesBuffer, nodeIndex, childIndex) & FILLED_BIT) != 0;
}

public bool nodeIsFilled(RWStructuredBuffer<SV64IndividualNode> nodesBuffer, uint nodeIndex, uint childIndex) {
    return (nodeChildData(nodesBuffer, nodeIndex, childIndex) & FILLED_BIT) != 0;
}
