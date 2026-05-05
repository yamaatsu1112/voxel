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
    if (!inCommandRange(threadIndex)) {
        return;
    }

    initializeCompactAllDepthMaterialized(
        svoNodes[pushConstants.svoIndex],
        svoLeaves[pushConstants.svoIndex],
        svoFreeNodeIndices[pushConstants.svoIndex],
        svoFreeLeafIndices[pushConstants.svoIndex],
        materializeRequests[pushConstants.svoIndex],
        prefixSums[pushConstants.svoIndex],
        flags[pushConstants.svoIndex],
        editWorkspace[pushConstants.svoIndex][1],
        editWorkspace[pushConstants.svoIndex][2],
        editWorkspace[pushConstants.svoIndex][3],
        editWorkspace[pushConstants.svoIndex][4],
        threadIndex
    );
}
