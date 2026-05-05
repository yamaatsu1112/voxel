module edit_bindings;

import svo;
import utils;
import prefix_sum;
import edit_allocation_common;
import edit_allocation_scan;
import edit_allocation_reset;
import edit_allocation_compact_all_depth;
import edit_allocation_compact_all_depth_materialize;
import edit_allocation_compact_all_depth_link;
import edit_allocation_apply;
import edit_allocation_collapse;

public const static uint NUM_SVO_DATA = 9;

[[vk::binding(0, 0)]]
public RWStructuredBuffer<SVONode> svoNodes[NUM_SVO_DATA];

[[vk::binding(1, 0)]]
public RWStructuredBuffer<SVOLeaf> svoLeaves[NUM_SVO_DATA];

[[vk::binding(2, 0)]]
public RWStructuredBuffer<SVOCounters> svoCounters[NUM_SVO_DATA];

[[vk::binding(3, 0)]]
public RWStructuredBuffer<uint> svoFreeNodeIndices[NUM_SVO_DATA];

[[vk::binding(4, 0)]]
public RWStructuredBuffer<uint> svoFreeLeafIndices[NUM_SVO_DATA];

[[vk::binding(5, 0)]]
public StructuredBuffer<LeafEditCommand> leafEditCommands[NUM_SVO_DATA];

[[vk::binding(6, 0)]]
public RWStructuredBuffer<NodeRequest> materializeRequests[NUM_SVO_DATA];

[[vk::binding(7, 0)]]
public RWStructuredBuffer<NodeRequest> compactRequests[NUM_SVO_DATA];

[[vk::binding(8, 0)]]
public RWStructuredBuffer<uint> flags[NUM_SVO_DATA];

[[vk::binding(9, 0)]]
public RWStructuredBuffer<uint> prefixSums[NUM_SVO_DATA];

[[vk::binding(10, 0)]]
public RWStructuredBuffer<uint> editWorkspace[NUM_SVO_DATA];

[[vk::binding(11, 0)]]
public ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(12, 0)]]
public StructuredBuffer<uint> commandCounts;

public struct EditPushConstants {
    public uint depth;
    public uint unusedLayerType;
    public uint phase;
    public uint svoIndex;
};

[[vk::push_constant]]
public ConstantBuffer<EditPushConstants> pushConstants;

public uint commandCount() {
    return commandCounts[pushConstants.svoIndex];
}

public uint paddedCommandCount() {
    return nextPowerOf2(commandCount());
}

public bool inCommandRange(uint threadIndex) {
    return threadIndex < commandCount();
}

public bool inPaddedRange(uint threadIndex) {
    return threadIndex < paddedCommandCount();
}

public uint inclusiveFlagCount() {
    uint count = commandCount();
    return count == 0u ? 0u : prefixSums[pushConstants.svoIndex][count - 1u];
}

public void snapshotAllocationState(uint threadIndex) {
    if (threadIndex != 0u) {
        return;
    }

    uint svoIndex = pushConstants.svoIndex;
    editWorkspace[svoIndex][1] = svoCounters[svoIndex][0].nodeCount;
    editWorkspace[svoIndex][2] = svoCounters[svoIndex][0].leafCount;
    editWorkspace[svoIndex][3] = svoCounters[svoIndex][0].freeNodeCount;
    editWorkspace[svoIndex][4] = svoCounters[svoIndex][0].freeLeafCount;
}
