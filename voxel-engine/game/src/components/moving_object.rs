use voxel_engine::engine_types::Vec3;
use voxel_engine::Component;

/// Moving object marker component with velocity
#[derive(Component, Clone, Copy)]
pub struct MovingObject {
    pub velocity: Vec3,
}
