implementing svo;

public static const uint TILE_SIZE = 8u;
public static const uint NUM_TILES = 256u;

public float3 lookupVoxelColor(
    Texture3D<float4> voxelTexture3D,
    SamplerState voxelTexture3DSampler,
    uint voxelId,
    int3 voxelCoord
) {
    float u = float(((voxelCoord.x % int(TILE_SIZE)) + int(TILE_SIZE)) % int(TILE_SIZE)) + 0.5;
    float v = float(((voxelCoord.y % int(TILE_SIZE)) + int(TILE_SIZE)) % int(TILE_SIZE)) + 0.5;
    float w =
        float(
            voxelId * TILE_SIZE +
            uint(((voxelCoord.z % int(TILE_SIZE)) + int(TILE_SIZE)) % int(TILE_SIZE))
        ) + 0.5;

    float3 texCoord = float3(
        u / float(TILE_SIZE),
        v / float(TILE_SIZE),
        w / float(TILE_SIZE * NUM_TILES)
    );

    return voxelTexture3D.Sample(voxelTexture3DSampler, texCoord).rgb;
}

public float4 computeFrag3DTexColor<T : ISVOAccessor>(
    T svo,
    UniformBufferObject ubo,
    SelectedGBufferHit selected,
    uint voxelId,
    Texture3D<float4> voxelTexture3D,
    SamplerState voxelTexture3DSampler,
    float2 pixelCoord
) {
    float4 color = float4(0.02, 0.02, 0.08, 1.0);
    if (selected.faceIndex == uint(Face::NO_HIT)) {
        return color;
    }

    int3 voxelCoord = int3(
        selected.gbufferData.x,
        selected.gbufferData.y,
        selected.gbufferData.z
    );
    float3 voxelColor = lookupVoxelColor(
        voxelTexture3D,
        voxelTexture3DSampler,
        voxelId,
        voxelCoord
    );
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
