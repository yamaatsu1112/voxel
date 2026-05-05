// G-Buffer Color Write Fragment Shader - Slang version
// Writes voxel color bits to G-Buffer using SVO traversal

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

[[vk::binding(3, 0)]]
StructuredBuffer<SVONode> colorSvoNodesBuffers[24];

[[vk::binding(4, 0)]]
StructuredBuffer<SVOLeaf> colorSvoLeavesBuffers[24];

[[vk::binding(5, 0)]]
Texture2D<uint4> gbufferImage;
[[vk::binding(5, 0)]]
SamplerState gbufferImageState;

// ============================================================================
// Push Constants
// ============================================================================

[[vk::push_constant]]
cbuffer PushConstants {
    uint bitIndex;
};

// ============================================================================
// Fragment Shader Entry Point
// ============================================================================

struct FragmentInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD;
};

struct FragmentOutput {
    [[vk::location(0)]]
    uint4 color : SV_Target0;  // Single color channel (R, G, B, A all same)
};

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    uint4 gbufferData = gbufferImage.Sample(gbufferImageState, input.texCoord);
    SVOAccessor svo;
    svo.nodesBuffer = colorSvoNodesBuffers[bitIndex];
    svo.leavesBuffer = colorSvoLeavesBuffers[bitIndex];
    FragmentOutput output;
    output.color = computeGBufferColorWrite(svo, gbufferData, bitIndex);
    return output;
}
