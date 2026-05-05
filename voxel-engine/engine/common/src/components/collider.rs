use crate::ecs::component::Component;
use engine_app::engine_types::Vec3;
use engine_macro::Component;

#[derive(Component, Clone, Copy)]
pub struct Collider {
    pub half_extent: Vec3,
}
