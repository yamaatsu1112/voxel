implementing svo;

public interface ISVOAccessor {
    RaycastResult raycast(float3 origin, float3 direction);
    bool raycastAnyHit(float3 origin, float3 direction);
    bool getVoxel(uint x, uint y, uint z);
}

public bool getVoxel(
    StructuredBuffer<SVONode> nodesBuffer,
    StructuredBuffer<SVOLeaf> leavesBuffer,
    uint x,
    uint y,
    uint z
) {
    uint nodeIndex = ROOT_NODE_INDEX;

    for (uint depth = 0; depth < MAX_DEPTH - 1; depth++) {
        uint childIndex = getChildIndex(x, y, z, depth);

        if (!nodeHasChild(nodesBuffer, nodeIndex, childIndex)) {
            return nodeIsFilled(nodesBuffer, nodeIndex, childIndex);
        }
        nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, childIndex);
    }

    uint childIndex = getChildIndex(x, y, z, MAX_DEPTH - 1);
    if (!nodeHasChild(nodesBuffer, nodeIndex, childIndex)) {
        return nodeIsFilled(nodesBuffer, nodeIndex, childIndex);
    }
    nodeIndex = nodeChildIndex(nodesBuffer, nodeIndex, childIndex);

    uint bitOffset = getLeafOffset(x, y, z);
    return leavesBuffer[nodeIndex].getVoxel(bitOffset);
}
