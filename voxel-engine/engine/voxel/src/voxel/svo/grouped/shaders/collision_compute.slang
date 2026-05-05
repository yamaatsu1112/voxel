import svo;

[[vk::binding(0, 0)]]
StructuredBuffer<float> collisionInput;

[[vk::binding(1, 0)]]
StructuredBuffer<SVONode> svoNodesBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<SVOLeaf> svoLeavesBuffer;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> collisionResult;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    float3 pos = float3(collisionInput[0], collisionInput[1], collisionInput[2]);
    float3 halfExt = float3(collisionInput[3], collisionInput[4], collisionInput[5]);
    float3 velocity = float3(collisionInput[6], collisionInput[7], collisionInput[8]);
    SVOAccessor svo;
    svo.nodesBuffer = svoNodesBuffer;
    svo.leavesBuffer = svoLeavesBuffer;
    CollisionResultData result = computeCollisionResult(svo, pos, halfExt, velocity);
    collisionResult[0] = result.position.x;
    collisionResult[1] = result.position.y;
    collisionResult[2] = result.position.z;
    collisionResult[3] = result.collided.x;
    collisionResult[4] = result.collided.y;
    collisionResult[5] = result.collided.z;
}
