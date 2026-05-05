pub mod builders;
pub mod recordables;

pub use engine_common::ecs;
pub use engine_common::game_loop;
pub use engine_common::input_event_queue;
pub use engine_common::rendering;
pub use engine_common::vulkan;
pub use engine_common::window;

pub use engine_common::buffer_configs;
pub use engine_common::descriptor_set_configs;
pub use engine_common::recordables as common_recordables;
pub use engine_common::uniform_buffer_object;
pub use engine_ui::builders as ui_builders;
pub use engine_ui::recordables as ui_recordables;
pub use engine_ui::systems as ui_systems;
pub use engine_voxel::builders as voxel_builders;
pub use engine_voxel::recordables as voxel_recordables;
pub use engine_voxel::systems as voxel_systems;
pub use engine_voxel::vertex;
pub use engine_voxel::voxel;

pub use engine_common::Camera;
pub use engine_common::CameraRenderData;
pub use engine_common::CameraRenderItem;
pub use engine_common::CameraTransformData;
pub use engine_common::Collider;
pub use engine_common::LocalTransform;
pub use engine_common::Parent;
pub use engine_common::Transform;
pub use engine_common::WorldLinkAllocator;
pub use engine_common::WorldLinkId;
pub use engine_common::ensure_world_link_id;
pub use engine_ui::Button;
pub use engine_ui::UI;
pub use engine_ui::UIInstance;
pub use engine_voxel::DynamicVoxelObject;

pub use engine_voxel::builders::configs::buffer_configs::CollisionInputBuffer;
pub use engine_voxel::builders::configs::buffer_configs::CollisionResultBuffer;

pub use engine_voxel::CollisionComputePipelineRecordable;
pub use engine_voxel::CollisionComputeSubmitGroup;
pub use engine_voxel::DynamicVoxelComputeSubmitGroup;
pub use engine_voxel::DynamicVoxelPipelineMutex;
pub use engine_voxel::DynamicVoxelPipelineRecordable;
pub use engine_voxel::GraphicsTransferMutex;
pub use engine_voxel::RaycastComputePipelineRecordable;
pub use engine_voxel::RaycastComputeSubmitGroup;
pub use engine_voxel::VoxelComputeSubmitGroup;
pub use engine_voxel::VoxelDestroyPipelineRecordable;
pub use engine_voxel::VoxelPipelineMutex;
pub use engine_voxel::VoxelPipelineRecordable;

pub fn setup(engine: &mut engine_app::VoxelEngine) {
    engine_common::setup(engine);
    engine.add_system_once(create_descriptor_pool_system);
    engine_voxel::setup(engine);
    engine_ui::setup(engine);
    engine.add_render_last_system(recordables::construct_render_graph_system);
}

fn create_descriptor_pool_system(world: &mut engine_app::World) {
    let mut renderer = world.get_resource_mut::<engine_app::Renderer>().unwrap();
    builders::create_descriptor_pool(&mut renderer);
}
