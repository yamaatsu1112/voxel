use voxel_engine::*;

use crate::components::{CollisionState, Player, Velocity};

/// Jump initial velocity
pub const JUMP_VELOCITY: f32 = 8.0;

pub fn jump_system(world: &mut World) {
    let mut query = world.query::<(&mut Velocity, &CollisionState, &Player)>();
    let input_state = world.get_resource::<InputState>().unwrap();

    for (_, (velocity, collision_state, _player)) in query.iter_mut() {
        if collision_state.collided_y && input_state.is_key_just_pressed(Key::KeySpace) {
            velocity.velocity.y = JUMP_VELOCITY;
        }
    }
}
