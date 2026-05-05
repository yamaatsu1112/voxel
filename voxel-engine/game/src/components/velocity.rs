use voxel_engine::engine_types::Vec3;
use voxel_engine::Component;

/// Generic velocity component in units/second.
#[derive(Component, Clone, Copy)]
pub struct Velocity {
    pub velocity: Vec3,
}

impl Default for Velocity {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_default() {
        let velocity = Velocity::default();
        assert_eq!(velocity.velocity.x, 0.0);
        assert_eq!(velocity.velocity.y, 0.0);
        assert_eq!(velocity.velocity.z, 0.0);
    }
}
