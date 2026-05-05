import edit_bindings;
import svo;
import edit_allocation_common;
import edit_allocation_scan;
import edit_allocation_reset;
import edit_allocation_compact_all_depth;
import edit_allocation_compact_all_depth_materialize;
import edit_allocation_compact_all_depth_link;
import edit_allocation_apply;
import edit_allocation_collapse;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint threadIndex = dispatchThreadID.x;
    if (threadIndex != 0u) {
        return;
    }
    uint count = commandCount();
    if (count == 0u) {
        return;
    }
    uint nodeCount = prefixSums[pushConstants.svoIndex][count - 1u];
    uint leafCount = flags[pushConstants.svoIndex][count - 1u];
    uint baseFreeNodeCount = editWorkspace[pushConstants.svoIndex][3];
    uint baseFreeLeafCount = editWorkspace[pushConstants.svoIndex][4];
    uint reusedNodes = clamp(nodeCount, 0u, baseFreeNodeCount);
    uint reusedLeaves = clamp(leafCount, 0u, baseFreeLeafCount);

    svoCounters[pushConstants.svoIndex][0].nodeCount =
        editWorkspace[pushConstants.svoIndex][1] + (nodeCount - reusedNodes);
    svoCounters[pushConstants.svoIndex][0].leafCount =
        editWorkspace[pushConstants.svoIndex][2] + (leafCount - reusedLeaves);
    svoCounters[pushConstants.svoIndex][0].freeNodeCount = baseFreeNodeCount - reusedNodes;
    svoCounters[pushConstants.svoIndex][0].freeLeafCount = baseFreeLeafCount - reusedLeaves;
}
