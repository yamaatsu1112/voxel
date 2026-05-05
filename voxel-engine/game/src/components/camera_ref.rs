use voxel_engine::{Component, Entity};

/// Component that holds a reference to the camera entity.
/// Added to the player entity to allow systems to access camera data.
#[derive(Component, Clone, Copy)]
pub struct CameraRef {
    pub entity: Entity,
}
