implementing svo;

public struct GBufferWriteResult {
    public uint4 gbuffer;
    public float depth;
}

public GBufferWriteResult computeGBufferWrite<T : ISVOAccessor>(
    T svo,
    UniformBufferObject ubo,
    float2 pixelCoord
) {
    float3 rayOrigin = ubo.cameraPosition;
    float3 rayDirection = calculateRayDirection(ubo, pixelCoord, rayOrigin);
    RaycastResult result = svo.raycast(rayOrigin, rayDirection);

    GBufferWriteResult output;
    if (result.face != Face::NO_HIT) {
        output.gbuffer = uint4(
            result.voxelCoord.x,
            result.voxelCoord.y,
            result.voxelCoord.z,
            uint(result.face)
        );
        output.depth = result.tc_min;
    } else {
        output.gbuffer = uint4(0, 0, 0, uint(Face::NO_HIT));
        output.depth = 0.0;
    }

    return output;
}
