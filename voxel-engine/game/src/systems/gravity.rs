use voxel_engine::*;

use crate::components::{Player, Velocity};

/// Gravity constant (game-tuned value for snappy action feel)
pub const GRAVITY: f32 = 25.0;

pub fn gravity_system(world: &mut World) {
    let mut query = world.query::<(&mut Velocity, &Player)>();
    let dt = world.get_resource::<DeltaTime>().unwrap();

    for (_, (velocity, _player)) in query.iter_mut() {
        velocity.velocity.y -= GRAVITY * dt.dt as f32;
    }
}
