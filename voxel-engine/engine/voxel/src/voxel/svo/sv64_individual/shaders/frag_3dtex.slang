// Deferred Rendering Fragment Shader with 3D Texture - Slang version
// Reads from G-Buffer, resolves voxel ID to color via 3D texture, applies lighting/shadows

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
Texture2D<uint4> gbufferImage;
[[vk::binding(3, 0)]]
SamplerState gbufferImageState;

[[vk::binding(4, 0)]]
Texture2D<float> gbufferDepthImage;
[[vk::binding(4, 0)]]
SamplerState gbufferDepthImageState;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba8ui")]]
Texture2D<uint4> gbufferColorImage;
[[vk::binding(5, 0)]]
SamplerState gbufferColorImageState;

// Dynamic G-Buffer bindings
[[vk::binding(6, 0)]]
Texture2D<uint4> dynamicGbufferImage;
[[vk::binding(6, 0)]]
SamplerState dynamicGbufferImageState;

[[vk::binding(7, 0)]]
Texture2D<float> dynamicGbufferDepthImage;
[[vk::binding(7, 0)]]
SamplerState dynamicGbufferDepthImageState;

[[vk::binding(8, 0)]]
[[vk::image_format("rgba8ui")]]
Texture2D<uint4> dynamicGbufferColorImage;
[[vk::binding(8, 0)]]
SamplerState dynamicGbufferColorImageState;

// 3D texture for voxel ID -> color lookup
[[vk::binding(0, 1)]]
Texture3D<float4> voxelTexture3D;
[[vk::binding(0, 1)]]
SamplerState voxelTexture3DSampler;

struct FragmentInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD;
};

struct FragmentOutput {
    [[vk::location(0)]]
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    uint4 staticGbufferData = gbufferImage.Sample(gbufferImageState, input.texCoord);
    float staticDepth = gbufferDepthImage.Sample(gbufferDepthImageState, input.texCoord);
    uint4 dynamicGbufferData = dynamicGbufferImage.Sample(dynamicGbufferImageState, input.texCoord);
    float dynamicDepth = dynamicGbufferDepthImage.Sample(dynamicGbufferDepthImageState, input.texCoord);
    SVOAccessor svo;
    svo.nodesBuffer = svoNodesBuffer;
    svo.leavesBuffer = svoLeavesBuffer;
    SelectedGBufferHit selected = selectClosestGBufferHit(
        staticGbufferData,
        staticDepth,
        dynamicGbufferData,
        dynamicDepth
    );

    FragmentOutput output;
    uint voxelId = 0u;
    if (selected.faceIndex != uint(Face::NO_HIT)) {
        uint4 idData = selected.useStatic ?
            gbufferColorImage.Sample(gbufferColorImageState, input.texCoord) :
            dynamicGbufferColorImage.Sample(dynamicGbufferColorImageState, input.texCoord);
        voxelId = idData.r;
    }
    output.color = computeFrag3DTexColor(
        svo,
        ubo,
        selected,
        voxelId,
        voxelTexture3D,
        voxelTexture3DSampler,
        input.position.xy
    );
    return output;
}
