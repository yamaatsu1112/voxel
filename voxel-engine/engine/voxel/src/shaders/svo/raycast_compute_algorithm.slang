implementing svo;

public struct RaycastResultData {
    uint3 voxelCoord;
    uint face;
    float tc_min;
}

public RaycastResultData computeRaycastResultData<T : ISVOAccessor>(
    T svo,
    UniformBufferObject ubo
) {
    float3 rayOrigin = ubo.cameraPosition;
    float3 rayDirection = calculateCenterRayDirection(ubo, rayOrigin);
    RaycastResult result = svo.raycast(rayOrigin, rayDirection);

    RaycastResultData outputData;
    outputData.voxelCoord = result.voxelCoord;
    outputData.face = uint(result.face);
    outputData.tc_min = result.tc_min;
    return outputData;
}
