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
    snapshotAllocationState(threadIndex);
}
