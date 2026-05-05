use crate::ecs::component::Component;
use crate::rendering::interpolation::Interpolatable;
use engine_app::engine_types::Vec3;
use engine_macro::Component;

#[derive(Component, Clone, Copy, Default, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Vec3,
}

impl Interpolatable for Transform {
    fn interpolate(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position + (other.position - self.position) * t,
            rotation: self.rotation + (other.rotation - self.rotation) * t,
        }
    }
}
