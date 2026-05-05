use crate::builders::configs::buffer_configs::RaycastResultBuffer;
use crate::ecs::world::World;
use crate::rendering::executor::Executor;
use crate::voxel::{RawVoxelRaycastData, VoxelRaycastData, WORLD_VOXEL_COUNT};

/// Initialize VoxelRaycastData as a Resource.
/// This system runs once at startup.
pub fn set_voxel_raycast_data(world: &mut World) {
    world.insert_resource(VoxelRaycastData::default());
}

/// System that reads raycast results from the GPU buffer.
/// This system should run once per frame to update voxel raycast data.
/// The raycast compute shader is dispatched as part of dispatch_compute().
pub fn update_voxel_raycast_system(world: &mut World) {
    let new_raycast_data = {
        let compute = world.get_resource::<Executor>().unwrap();
        let Some(raw) = compute.read_buffer_data::<RaycastResultBuffer, u32, 5>(0) else {
            return;
        };
        let raw_data = RawVoxelRaycastData::from_buffer(&raw);

        let shifted_coord = if raw_data.is_hit() {
            (
                raw_data.voxel_coord.0 as i32 - (WORLD_VOXEL_COUNT as i32 >> 1),
                raw_data.voxel_coord.1 as i32 - (WORLD_VOXEL_COUNT as i32 >> 1),
                raw_data.voxel_coord.2 as i32 - (WORLD_VOXEL_COUNT as i32 >> 1),
            )
        } else {
            (i32::MAX, i32::MAX, i32::MAX)
        };

        VoxelRaycastData {
            voxel_coord: shifted_coord,
            distance: raw_data.distance,
            face: raw_data.face,
        }
    };

    let mut raycast_data = world.get_resource_mut::<VoxelRaycastData>().unwrap();
    *raycast_data = new_raycast_data;
}
