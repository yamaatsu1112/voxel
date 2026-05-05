// SVO Allocation Helpers - Shared helper functions for edit pipelines.

implementing svo;

public void prefixSumSharedLocal(
    RWStructuredBuffer<uint> prefixSumBuffer,
    RWStructuredBuffer<uint> requestCountBuffer,
    uint paddedSize,
    uint localIndex,
    uint groupIndex,
    uint threadIndex,
    inout uint sharedData[64]
) {
    sharedData[localIndex] = (threadIndex < paddedSize) ? prefixSumBuffer[threadIndex] : 0u;
    GroupMemoryBarrierWithGroupSync();

    inclusivePrefixSumShared<64, 1>(sharedData, localIndex, 64u);

    if (threadIndex < paddedSize) {
        prefixSumBuffer[threadIndex] = sharedData[localIndex];
    }

    if (localIndex == 63u) {
        requestCountBuffer[1u + groupIndex] = sharedData[63u];
    }
}

public void prefixSumSharedBlockScan(
    RWStructuredBuffer<uint> requestCountBuffer,
    uint numWG,
    uint localIndex,
    inout uint sharedData[64]
) {
    sharedData[localIndex] = (localIndex < numWG) ? requestCountBuffer[1u + localIndex] : 0u;
    GroupMemoryBarrierWithGroupSync();

    inclusivePrefixSumShared<64, 1>(sharedData, localIndex, 64u);

    if (localIndex < numWG) {
        requestCountBuffer[1u + localIndex] = sharedData[localIndex];
    }
}

public void prefixSumSharedPropagate(
    RWStructuredBuffer<uint> prefixSumBuffer,
    RWStructuredBuffer<uint> requestCountBuffer,
    uint paddedSize,
    uint localIndex,
    uint groupIndex,
    uint threadIndex
) {
    if (groupIndex > 0u && threadIndex < paddedSize) {
        prefixSumBuffer[threadIndex] += requestCountBuffer[groupIndex];
    }
}
