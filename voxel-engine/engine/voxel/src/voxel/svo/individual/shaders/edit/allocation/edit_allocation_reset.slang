module edit_allocation_reset;

import svo;

public void resetIndividualSVO(
    RWStructuredBuffer<SVOIndividualNode> nodesBuffer,
    RWStructuredBuffer<SVOCounters> countersBuffer
) {
    for (uint i = 0u; i < 8u; i++) {
        nodesBuffer[ROOT_NODE_INDEX].childData[i] = 0u;
    }
    countersBuffer[0].nodeCount = 1u;
    countersBuffer[0].leafCount = 0u;
    countersBuffer[0].freeNodeCount = 0u;
    countersBuffer[0].freeLeafCount = 0u;
}
