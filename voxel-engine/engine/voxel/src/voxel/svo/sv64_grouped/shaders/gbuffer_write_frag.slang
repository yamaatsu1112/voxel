// G-Buffer Write Fragment Shader - Slang version
// Writes voxel raycast results to G-Buffer

import svo;

// ============================================================================
// Resource Bindings (Vulkan-style)
// ============================================================================

[[vk::binding(0, 0)]]
 ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(1, 0)]]
 StructuredBuffer<SVONode> svoNodesBuffer;

[[vk::binding(2, 0)]]
 StructuredBuffer<SVOLeaf> svoLeavesBuffer;

struct FragmentInput {
    float4 position : SV_Position;
};

struct FragmentOutput {
    [[vk::location(0)]]
    uint4 gbuffer : SV_Target0;  // [voxel_x, voxel_y, voxel_z, face]
    [[vk::location(1)]]
    float depth : SV_Target1;    // tc_min depth
};

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    SVOAccessor svo;
    svo.nodesBuffer = svoNodesBuffer;
    svo.leavesBuffer = svoLeavesBuffer;
    GBufferWriteResult result = computeGBufferWrite(svo, ubo, input.position.xy);
    FragmentOutput output;
    output.gbuffer = result.gbuffer;
    output.depth = result.depth;
    return output;
}
