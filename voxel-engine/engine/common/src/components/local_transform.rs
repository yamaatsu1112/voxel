use crate::ecs::component::Component;
use engine_app::engine_types::Vec3;
use engine_macro::Component;

#[derive(Component, Clone, Copy, Default, Debug)]
pub struct LocalTransform {
    pub position: Vec3,
}

impl LocalTransform {
    pub fn new(position: Vec3) -> Self {
        Self { position }
    }
}
