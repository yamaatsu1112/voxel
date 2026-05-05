pub mod buffer_configs;
pub mod builders;
pub mod components;
pub mod descriptor_set_configs;
pub mod recordables;
pub mod systems;
pub mod uniform_buffer_object;

pub mod ecs {
    pub use engine_app::ecs::*;
}

pub mod game_loop {
    pub use engine_app::game_loop::*;
}

pub mod input_event_queue {
    pub use engine_app::input_event_queue::*;
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

pub use buffer_configs::uniform::{ComputeUniformBuffer, UniformBuffer};
pub use components::camera::Camera;
pub use components::collider::Collider;
pub use components::local_transform::LocalTransform;
pub use components::parent::Parent;
pub use components::transform::Transform;
pub use descriptor_set_configs::pools::EngineDescriptorPool;
pub use engine_app::{WorldLinkAllocator, WorldLinkId, ensure_world_link_id};
pub use systems::camera::CameraTransformData;
pub use systems::render_world_camera::{CameraRenderData, CameraRenderItem};
pub use uniform_buffer_object::UniformBufferObject;

pub fn setup(engine: &mut engine_app::VoxelEngine) {
    engine.add_system_once(setup_common_renderer_system);
    engine.add_system_once(setup_world_link_allocator_system);
    engine.add_system_once(systems::input::set_input_state);
    engine.add_system_once(systems::window_size::set_window_size);

    engine.add_first_system(systems::frame_system::frame_begin_system);
    engine.add_first_system(systems::camera::game_camera_update_system);
    engine.add_first_system(systems::input::input_system);
    engine.add_first_system(systems::window_size::update_window_size);

    engine.add_last_system(systems::local_transform_system::local_transform_system);
    engine.add_last_system(systems::frame_system::frame_end_system);

    engine.add_render_system_once(systems::input::set_input_state);
    engine.add_render_system_once(
        systems::render_world_transform_interpolation::setup_render_world_interpolation_system,
    );
    engine.add_render_system_once(systems::render_world_camera::setup_render_camera_system);
    engine.add_render_system_once(systems::window_size::set_window_size);

    engine.add_render_first_system(systems::frame_system::render_frame_begin_system);
    engine.add_render_first_system(systems::render_input::render_input_system);
    engine.add_render_first_system(systems::window_size::update_window_size);
    engine.add_render_first_system(
        systems::render_world_transform_interpolation::receive_camera_render_data_system,
    );

    engine
        .add_render_last_system(systems::render_world_transform_interpolation::interpolate_system);
    engine.add_render_last_system(systems::render_world_camera::render_camera_system);
    engine.add_render_last_system(systems::frame_system::render_frame_end_system);
}

fn setup_common_renderer_system(world: &mut engine_app::World) {
    let mut renderer = world.get_resource_mut::<engine_app::Renderer>().unwrap();
    builders::setup_common_renderer(&mut renderer);
}

fn setup_world_link_allocator_system(world: &mut engine_app::World) {
    world.insert_resource(engine_app::WorldLinkAllocator::default());
}
