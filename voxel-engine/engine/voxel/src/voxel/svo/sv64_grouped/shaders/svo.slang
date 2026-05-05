// SVO (Sparse Voxel Octree) Common Definitions - Slang Module
// This module contains shared data structures, constants, and helper functions
// used by fragment and compute shaders.

module svo;

import utils;
import prefix_sum;

__include "svo/types.slang";
__include "types.slang";
public typealias SVONode = SVOGroupedNode;
__include "svo/face.slang";
__include "uniform_buffer_object.slang";
__include "svo/helpers.slang";
__include "raycast.slang";
__include "svo/svo_accessor.slang";
public struct SVOAccessor : ISVOAccessor {
    public StructuredBuffer<SVONode> nodesBuffer;
    public StructuredBuffer<SVOLeaf> leavesBuffer;

    public RaycastResult raycast(float3 origin, float3 direction) {
        return SVO.raycast(this.nodesBuffer, this.leavesBuffer, origin, direction);
    }

    public bool raycastAnyHit(float3 origin, float3 direction) {
        return SVO.raycastAnyHit(this.nodesBuffer, this.leavesBuffer, origin, direction);
    }

    public bool getVoxel(uint x, uint y, uint z) {
        return getVoxel(this.nodesBuffer, this.leavesBuffer, x, y, z);
    }
}
__include "svo/gbuffer_write_algorithm.slang";
__include "svo/gbuffer_id_write_algorithm.slang";
__include "svo/gbuffer_color_write_algorithm.slang";
__include "svo/frag_algorithm.slang";
__include "svo/frag_3dtex_algorithm.slang";
__include "svo/raycast_compute_algorithm.slang";
__include "svo/collision_compute_algorithm.slang";
__include "svo/leaf_edit_policy.slang";
__include "svo/allocation_helpers.slang";
__include "allocation.slang";
