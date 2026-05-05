use crate::ecs::component::Component;
use engine_app::engine_types::Vec2;
use engine_macro::Component;

#[derive(Component, Clone, Copy, Default, Debug)]
pub struct UI {
    pub position_x: u32,
    pub position_y: u32,
    pub width: u32,
    pub height: u32,
    pub atlas_offset: Vec2,
    pub atlas_size: Vec2,
}

#[repr(C)]
#[derive(Clone)]
pub struct UIInstance {
    pub position_x: f32,
    pub position_y: f32,
    pub width: f32,
    pub height: f32,
    pub atlas_u_offset: f32,
    pub atlas_v_offset: f32,
    pub atlas_u_size: f32,
    pub atlas_v_size: f32,
}
