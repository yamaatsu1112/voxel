// Dynamic G-Buffer ID Write Fragment Shader - Slang version
// Writes dynamic voxel ID bits to Dynamic G-Buffer using SVO traversal

import svo;

// ============================================================================
// Resource Bindings (Vulkan-style)
// ============================================================================

[[vk::binding(0, 0)]]
 ConstantBuffer<UniformBufferObject> ubo;

// Dynamic SVO buffer (occupancy)
[[vk::binding(1, 0)]]
 StructuredBuffer<SVONode> dynamicSvoNodesBuffer;

[[vk::binding(2, 0)]]
 StructuredBuffer<SVOLeaf> dynamicSvoLeavesBuffer;

// Dynamic ID SVO buffers (8 ID bit planes)
[[vk::binding(3, 0)]]
 StructuredBuffer<SVONode> dynamicIdSvoNodesBuffers[8];

[[vk::binding(4, 0)]]
 StructuredBuffer<SVOLeaf> dynamicIdSvoLeavesBuffers[8];

// Dynamic G-Buffer image (voxel coordinates + face)
[[vk::binding(5, 0)]]
Texture2D<uint4> dynamicGbufferImage;
[[vk::binding(5, 0)]]
SamplerState dynamicGbufferImageState;

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
    uint4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    uint4 gbufferData = dynamicGbufferImage.Sample(dynamicGbufferImageState, input.texCoord);
    SVOAccessor svo;
    svo.nodesBuffer = dynamicIdSvoNodesBuffers[bitIndex];
    svo.leavesBuffer = dynamicIdSvoLeavesBuffers[bitIndex];
    FragmentOutput output;
    output.color = computeGBufferIdWrite(svo, gbufferData, bitIndex);
    return output;
}
