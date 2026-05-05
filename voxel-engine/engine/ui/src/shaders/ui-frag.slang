// UI Fragment Shader - Slang version

// ============================================================================
// Resource Bindings
// ============================================================================

[[vk::binding(0, 0)]]
Texture2D texSampler;

[[vk::binding(0, 0)]]
SamplerState texSamplerState;

// ============================================================================
// Fragment Input/Output Structures
// ============================================================================

struct FragmentInput {
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 texCoord : TEXCOORD;
};

struct FragmentOutput {
    [[vk::location(0)]]
    float4 color : SV_Target0;
};

// ============================================================================
// Fragment Shader Entry Point
// ============================================================================

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    FragmentOutput output;
    output.color = texSampler.Sample(texSamplerState, input.texCoord);
    return output;
}
