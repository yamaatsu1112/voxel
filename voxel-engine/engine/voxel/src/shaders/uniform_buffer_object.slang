// Uniform Buffer Object - Shared UBO Structure
// This module contains the UniformBufferObject structure used across shaders.

implementing svo;

// Uniform Buffer Object
// NOTE: This layout must match `UniformBufferObject` in Rust (uniform_buffer_object.rs)
public struct UniformBufferObject {
    public float4x4 view;
    public float4x4 proj;
    public float4x4 invViewProj;
    public float3 cameraPosition;
    public float _pad0;
    public float3 skyLightDirection;
    public float _pad1;
    public float2 windowSize;
};
