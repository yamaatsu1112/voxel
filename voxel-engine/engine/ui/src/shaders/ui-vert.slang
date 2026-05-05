// UI Vertex Shader - Slang version

// ============================================================================
// Vertex Input/Output Structures
// ============================================================================

struct VertexInput {
    [[vk::location(0)]]
    float positionX : POSITION_X;

    [[vk::location(1)]]
    float positionY : POSITION_Y;

    [[vk::location(2)]]
    float width : WIDTH;

    [[vk::location(3)]]
    float height : HEIGHT;

    [[vk::location(4)]]
    float atlasUOffset : ATLAS_U_OFFSET;

    [[vk::location(5)]]
    float atlasVOffset : ATLAS_V_OFFSET;

    [[vk::location(6)]]
    float atlasUSize : ATLAS_U_SIZE;

    [[vk::location(7)]]
    float atlasVSize : ATLAS_V_SIZE;
};

struct VertexOutput {
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 texCoord : TEXCOORD;
};

// ============================================================================
// Vertex Helper Data
// ============================================================================

struct Vertex {
    float2 position;
    float2 texCoord;
};

static const Vertex TOP_LEFT = { float2(-1.0, -1.0), float2(0.0, 0.0) };
static const Vertex TOP_RIGHT = { float2(1.0, -1.0), float2(1.0, 0.0) };
static const Vertex BOTTOM_RIGHT = { float2(1.0, 1.0), float2(1.0, 1.0) };
static const Vertex BOTTOM_LEFT = { float2(-1.0, 1.0), float2(0.0, 1.0) };

static const Vertex VERTICES[6] = {
    TOP_LEFT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    BOTTOM_RIGHT,
    TOP_RIGHT,
    TOP_LEFT
};

// ============================================================================
// Vertex Shader Entry Point
// ============================================================================

[shader("vertex")]
VertexOutput main(VertexInput input, uint vertexIndex : SV_VertexID) {
    VertexOutput output;

    // Calculate position based on vertex index
    float2 position = TOP_LEFT.position + float2(input.positionX, input.positionY);

    // Adjust Y position for bottom vertices
    if (vertexIndex == 1 || vertexIndex == 2 || vertexIndex == 3) {
        position.y += input.height;
    }

    // Adjust X position for right vertices
    if (vertexIndex == 2 || vertexIndex == 3 || vertexIndex == 4) {
        position.x += input.width;
    }

    output.position = float4(position, 0.0, 1.0);

    // Calculate texture coordinates using atlas offset and size
    float2 baseTexCoord = VERTICES[vertexIndex].texCoord;
    output.texCoord = float2(
        input.atlasUOffset + baseTexCoord.x * input.atlasUSize,
        input.atlasVOffset + baseTexCoord.y * input.atlasVSize
    );

    return output;
}
