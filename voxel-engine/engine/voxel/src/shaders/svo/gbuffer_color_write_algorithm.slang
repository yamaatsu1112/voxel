implementing svo;

public uint4 computeGBufferColorWrite<T : ISVOAccessor>(
    T svo,
    uint4 gbufferData,
    uint bitIndex
) {
    if (gbufferData.w == uint(Face::NO_HIT)) {
        return uint4(0u, 0u, 0u, 0u);
    }

    bool colorBit = svo.getVoxel(gbufferData.x, gbufferData.y, gbufferData.z);
    uint bitValue = colorBit ? (1u << (23 - bitIndex)) : 0u;
    return uint4(
        (bitValue & 0xFF0000) >> 16,
        (bitValue & 0x00FF00) >> 8,
        bitValue & 0x0000FF,
        0u
    );
}
