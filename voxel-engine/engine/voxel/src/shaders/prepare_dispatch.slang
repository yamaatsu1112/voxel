// Prepare Dispatch Shader
// Generates VkDispatchIndirectCommand for indirect dispatch

const static uint NUM_SVO_DATA = 9;
const static uint SLOTS_PER_SVO = 3;

[[vk::binding(0, 0)]]
StructuredBuffer<uint> commandCountBuffer;

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> indirectBuffer;

uint nextPowerOf2(uint n) {
    if (n == 0) return 0;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

[shader("compute")]
[numthreads(9, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint threadIndex = dispatchThreadID.x;
    if (threadIndex >= NUM_SVO_DATA) return;

    uint cmdCount = commandCountBuffer[threadIndex];
    uint base = threadIndex * SLOTS_PER_SVO * 3;

    if (cmdCount > 0) {
        uint padded = nextPowerOf2(cmdCount);
        indirectBuffer[base + 0] = (padded + 63) / 64;
        indirectBuffer[base + 1] = 1;
        indirectBuffer[base + 2] = 1;

        indirectBuffer[base + 3] = 1;
        indirectBuffer[base + 4] = 1;
        indirectBuffer[base + 5] = 1;

        indirectBuffer[base + 6] = (cmdCount + 63) / 64;
        indirectBuffer[base + 7] = 1;
        indirectBuffer[base + 8] = 1;
    } else {
        indirectBuffer[base + 0] = 0;
        indirectBuffer[base + 1] = 1;
        indirectBuffer[base + 2] = 1;

        indirectBuffer[base + 3] = 0;
        indirectBuffer[base + 4] = 1;
        indirectBuffer[base + 5] = 1;

        indirectBuffer[base + 6] = 0;
        indirectBuffer[base + 7] = 1;
        indirectBuffer[base + 8] = 1;
    }
}
