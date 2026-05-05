// Voxel Raycast Compute Shader
// Performs raycast from camera and stores voxel coordinate data

import svo;

// ============================================================================
// Resource Bindings
// ============================================================================

[[vk::binding(0, 0)]]
 ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(1, 0)]]
StructuredBuffer<SVONode> svoNodesBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<SVOLeaf> svoLeavesBuffer;

// Output buffer for raycast result (HOST_VISIBLE)
[[vk::binding(3, 0)]]
RWStructuredBuffer<RaycastResultData> raycastResultBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    SVOAccessor svo;
    svo.nodesBuffer = svoNodesBuffer;
    svo.leavesBuffer = svoLeavesBuffer;
    raycastResultBuffer[0] = computeRaycastResultData(svo, ubo);
}
