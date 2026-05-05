use crate::ecs::component::Component;
use engine_macro::Component;

#[derive(Component, Clone, Copy, Default, Debug)]
pub struct Camera {
    pub near: f32,
    pub far: f32,
    pub fov: f32,
}
