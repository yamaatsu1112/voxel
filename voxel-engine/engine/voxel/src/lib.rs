pub mod builders;
pub mod components;
pub mod recordables;
pub mod systems;
pub mod vertex;
pub mod voxel;

pub mod ecs {
    pub use engine_app::ecs::*;
}

pub mod game_loop {
    pub use engine_app::game_loop::*;
}

pub mod rendering {
    pub use engine_app::rendering::*;
}

pub mod vulkan {
    pub use engine_app::vulkan::*;
}

pub mod window {
    pub use engine_app::window::*;
}

pub use components::dynamic_voxel_object::DynamicVoxelObject;
pub use recordables::CollisionComputePipelineRecordable;
pub use recordables::CollisionComputeSubmitGroup;
pub use recordables::DynamicVoxelComputeSubmitGroup;
pub use recordables::DynamicVoxelPipelineMutex;
pub use recordables::DynamicVoxelPipelineRecordable;
pub use recordables::GraphicsTransferMutex;
pub use recordables::RaycastComputePipelineRecordable;
pub use recordables::RaycastComputeSubmitGroup;
pub use recordables::VoxelComputeSubmitGroup;
pub use recordables::VoxelDestroyPipelineRecordable;
pub use recordables::VoxelPipelineMutex;
pub use recordables::VoxelPipelineRecordable;
pub use voxel::CommandOffsetResource;
pub use voxel::DynamicVoxelObjectId;
pub use voxel::GpuCommandOffsets;
pub use voxel::NUM_SVO_DATA;
pub use voxel::VOXEL_SIZE;
pub use voxel::VoxelCommandBuffer;
pub use voxel::VoxelDestroyCommandBuffer;
pub use voxel::VoxelObjectData;
pub use voxel::VoxelObjectId;
pub use voxel::VoxelObjectManager;
pub use voxel::VoxelRaycastData;

pub fn setup(engine: &mut engine_app::VoxelEngine) {
    engine.add_system_once(register_voxel_resources);
    engine.add_system_once(setup_voxel_renderer_system);
    engine.add_system_once(systems::voxel_raycast::set_voxel_raycast_data);

    engine.add_last_system(systems::voxel_raycast::update_voxel_raycast_system);
    engine.add_last_system(systems::voxel_object_system::voxel_object_system);
    engine.add_last_system(systems::dynamic_voxel_object_system::dynamic_voxel_object_system);
    engine.add_last_system(systems::voxel_compute::voxel_compute_system);
}

fn register_voxel_resources(world: &mut engine_app::World) {
    world.insert_resource(voxel::VoxelObjectManager::new());
    world.insert_resource(voxel::VoxelCommandBuffer::new());
    world.insert_resource(voxel::VoxelDestroyCommandBuffer::new());
}

fn setup_voxel_renderer_system(world: &mut engine_app::World) {
    let mut renderer = world.get_resource_mut::<engine_app::Renderer>().unwrap();
    builders::setup_voxel_renderer(&mut renderer);
}
