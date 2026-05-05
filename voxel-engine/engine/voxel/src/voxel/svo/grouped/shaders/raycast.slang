// SVO Raycast wiring for grouped traversal.

implementing svo;

__include "raycast/common.slang";
__include "raycast/stack_impl.slang";
__include "raycast/root_retrace_impl.slang";
__include "raycast/policy.slang";

public extension SVO {
    public static RaycastResult raycast(
        StructuredBuffer<SVONode> nodesBuffer,
        StructuredBuffer<SVOLeaf> leavesBuffer,
        float3 rayOrigin,
        float3 rayDirection
    ) {
        return DefaultRaycastPolicy.raycast(
            nodesBuffer,
            leavesBuffer,
            rayOrigin,
            rayDirection
        );
    }

    public static bool raycastAnyHit(
        StructuredBuffer<SVONode> nodesBuffer,
        StructuredBuffer<SVOLeaf> leavesBuffer,
        float3 rayOrigin,
        float3 rayDirection
    ) {
        return DefaultRaycastPolicy.raycastAnyHit(
            nodesBuffer,
            leavesBuffer,
            rayOrigin,
            rayDirection
        );
    }
}
