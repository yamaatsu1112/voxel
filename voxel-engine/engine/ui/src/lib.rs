pub mod builders;
pub mod components;
pub mod recordables;
pub mod systems;

pub mod ecs {
    pub use engine_app::ecs::*;
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

pub use components::button::Button;
pub use components::ui::{UI, UIInstance};

pub fn setup(engine: &mut engine_app::VoxelEngine) {
    engine.add_system_once(setup_ui_renderer_system);
    engine.add_render_first_system(systems::button::button_input_system);
    engine.add_render_first_system(systems::button::button_callback_system);
}

fn setup_ui_renderer_system(world: &mut engine_app::World) {
    let mut renderer = world.get_resource_mut::<engine_app::Renderer>().unwrap();
    builders::setup_ui_renderer(&mut renderer);
}
