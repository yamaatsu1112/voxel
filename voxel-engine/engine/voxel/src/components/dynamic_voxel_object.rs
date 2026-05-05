use crate::ecs::component::Component;
use crate::voxel::DynamicVoxelObjectId;

/// DynamicVoxelObject component represents a rectangular prism of dynamic voxels.
/// Unlike VoxelObject, dynamic objects are expected to change every frame and
/// are always transferred to the GPU without dirty tracking.
///
/// When attached to an entity with a Transform component, the voxel data
/// will be rendered in the world at the transform's position.
///
/// This component stores only an ID to the actual voxel data,
/// which is managed by VoxelObjectManager.
#[derive(Debug, Clone, Copy)]
pub struct DynamicVoxelObject {
    pub id: DynamicVoxelObjectId,
}

impl Component for DynamicVoxelObject {}
