implementing svo;

public struct SelectedGBufferHit {
    public bool useStatic;
    public uint4 gbufferData;
    public float tcMin;
    public uint faceIndex;
}

public SelectedGBufferHit selectClosestGBufferHit(
    uint4 staticGbufferData,
    float staticDepth,
    uint4 dynamicGbufferData,
    float dynamicDepth
) {
    SelectedGBufferHit selected;

    bool staticHit = staticGbufferData.w != uint(Face::NO_HIT);
    bool dynamicHit = dynamicGbufferData.w != uint(Face::NO_HIT);

    if (dynamicHit && staticHit && dynamicDepth < staticDepth) {
        selected.useStatic = false;
        selected.gbufferData = dynamicGbufferData;
        selected.tcMin = dynamicDepth;
        selected.faceIndex = dynamicGbufferData.w;
        return selected;
    }

    selected.useStatic = true;
    selected.gbufferData = staticGbufferData;
    selected.tcMin = staticDepth;
    selected.faceIndex = staticGbufferData.w;

    if (dynamicHit && !staticHit) {
        selected.useStatic = false;
        selected.gbufferData = dynamicGbufferData;
        selected.tcMin = dynamicDepth;
        selected.faceIndex = dynamicGbufferData.w;
    }

    return selected;
}

public float3 shadeVoxelColor<T : ISVOAccessor>(
    T svo,
    UniformBufferObject ubo,
    float2 pixelCoord,
    uint faceIndex,
    float tcMin,
    float3 voxelColor
) {
    float3 surfaceNormal = getSurfaceNormal(Face(faceIndex));
    float3 rayOrigin = ubo.cameraPosition;
    float3 rayDirection = calculateRayDirection(ubo, pixelCoord, rayOrigin);
    float3 surfacePos =
        rayOrigin +
        tcMin * float(WORLD_VOXEL_COUNT >> VOXEL_SIZE_NEG_EXP) * rayDirection;
    float3 skyLightDir = normalize(ubo.skyLightDirection);
    surfacePos += 0.0001 * skyLightDir * sign(dot(skyLightDir, surfaceNormal));

    float diffuse = max(0.0, dot(skyLightDir, surfaceNormal));
    float3 ambient = float3(0.2, 0.2, 0.2);
    bool shadowHit = svo.raycastAnyHit(surfacePos, skyLightDir);
    float shadow = shadowHit ? 0.05 : 1.0;

    return voxelColor * (ambient + diffuse * shadow);
}

public float4 computeFragColor<T : ISVOAccessor>(
    T svo,
    UniformBufferObject ubo,
    SelectedGBufferHit selected,
    float3 voxelColor,
    float2 pixelCoord
) {
    float4 color = float4(0.02, 0.02, 0.08, 1.0);
    if (selected.faceIndex == uint(Face::NO_HIT)) {
        return color;
    }

    color.rgb = shadeVoxelColor(
        svo,
        ubo,
        pixelCoord,
        selected.faceIndex,
        selected.tcMin,
        voxelColor
    );
    return color;
}
