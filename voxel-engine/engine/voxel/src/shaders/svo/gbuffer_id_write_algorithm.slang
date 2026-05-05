implementing svo;

public uint4 computeGBufferIdWrite<T : ISVOAccessor>(
    T svo,
    uint4 gbufferData,
    uint bitIndex
) {
    if (gbufferData.w == uint(Face::NO_HIT)) {
        return uint4(0u, 0u, 0u, 0u);
    }

    bool idBit = svo.getVoxel(gbufferData.x, gbufferData.y, gbufferData.z);
    uint bitValue = idBit ? (1u << (7 - bitIndex)) : 0u;
    return uint4(bitValue, 0u, 0u, 0u);
}
