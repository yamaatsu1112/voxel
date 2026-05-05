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
    uint uniqueCount = inclusiveFlagCount();
    if (threadIndex >= uniqueCount) {
        return;
    }

    if (pushConstants.depth == MAX_DEPTH) {
        freeIndividualLeavesBatch(
            svoNodes[pushConstants.svoIndex],
            svoLeaves[pushConstants.svoIndex],
            svoFreeLeafIndices[pushConstants.svoIndex],
            compactRequests[pushConstants.svoIndex],
            editWorkspace[pushConstants.svoIndex][4],
            threadIndex
        );
    } else {
        freeIndividualNodesBatch(
            svoNodes[pushConstants.svoIndex],
            svoFreeNodeIndices[pushConstants.svoIndex],
            compactRequests[pushConstants.svoIndex],
            editWorkspace[pushConstants.svoIndex][3],
            threadIndex
        );
    }

    if (threadIndex == 0u) {
        if (pushConstants.depth == MAX_DEPTH) {
            svoCounters[pushConstants.svoIndex][0].freeLeafCount =
                editWorkspace[pushConstants.svoIndex][4] + uniqueCount;
        } else {
            svoCounters[pushConstants.svoIndex][0].freeNodeCount =
                editWorkspace[pushConstants.svoIndex][3] + uniqueCount;
        }
    }
}
