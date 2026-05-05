use crate::rendering::executor::Executor;
use crate::{
    CollisionComputeSubmitGroup, DynamicVoxelComputeSubmitGroup, RaycastComputeSubmitGroup,
    VoxelComputeSubmitGroup,
};

/// System that dispatches all compute shaders individually.
pub fn voxel_compute_system(world: &mut crate::ecs::world::World) {
    let mut compute = world.get_resource_mut::<Executor>().unwrap();
    compute.dispatch_compute::<VoxelComputeSubmitGroup>();
    compute.dispatch_compute::<DynamicVoxelComputeSubmitGroup>();
    compute.dispatch_compute::<RaycastComputeSubmitGroup>();
    compute.dispatch_compute::<CollisionComputeSubmitGroup>();
}
