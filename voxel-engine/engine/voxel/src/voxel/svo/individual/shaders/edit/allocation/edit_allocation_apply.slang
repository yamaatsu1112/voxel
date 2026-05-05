module edit_allocation_apply;

import svo;

public void updateIndividualLeafValues<P : ILeafEditPolicy>(
    RWStructuredBuffer<SVOIndividualNode> nodesBuffer,
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
