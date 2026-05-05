use voxel_engine::Component;

#[derive(Component, Clone, Copy, Default)]
pub struct CollisionState {
    pub collided_x: bool,
    pub collided_y: bool,
    pub collided_z: bool,
    pub has_collision_input: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collision_state_default() {
        let state = CollisionState::default();
        assert!(!state.collided_x);
        assert!(!state.collided_y);
        assert!(!state.collided_z);
        assert!(!state.has_collision_input);
    }
}
