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

groupshared uint scanSharedData[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint threadIndex = dispatchThreadID.x;
    uint localIndex = threadIndex & 63u;
    uint groupIndex = threadIndex >> 6u;
    uint paddedSize = paddedCommandCount();
    uint numWG = max(1u, paddedSize / 64u);

    if (pushConstants.phase == 0u) {
        if (threadIndex >= numWG * 64u) {
            return;
        }
        prefixSumSharedLocal(
            flags[pushConstants.svoIndex],
            editWorkspace[pushConstants.svoIndex],
            paddedSize,
            localIndex,
            groupIndex,
            threadIndex,
            scanSharedData
        );
    } else if (pushConstants.phase == 1u) {
        if (threadIndex >= 64u) {
            return;
        }
        prefixSumSharedBlockScan(
            editWorkspace[pushConstants.svoIndex],
            numWG,
            localIndex,
            scanSharedData
        );
    } else {
        if (threadIndex >= numWG * 64u) {
            return;
        }
        prefixSumSharedPropagate(
            flags[pushConstants.svoIndex],
            editWorkspace[pushConstants.svoIndex],
            paddedSize,
            localIndex,
            groupIndex,
            threadIndex
        );
    }
}
