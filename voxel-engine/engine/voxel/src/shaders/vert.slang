// Main Vertex Shader - Slang version

// ============================================================================
// Vertex Input/Output Structures
// ============================================================================

struct VertexInput {
    [[vk::location(0)]]
    float2 position : POSITION;

    [[vk::location(1)]]
    float2 texCoord : TEXCOORD;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD;
};

// ============================================================================
// Vertex Shader Entry Point
// ============================================================================

[shader("vertex")]
VertexOutput main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.texCoord = input.texCoord;
    return output;
}

