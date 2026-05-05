use voxel_engine::*;

use crate::components::MovingObject;

/// System to update positions of moving objects
pub fn moving_object_system(world: &mut World) {
    let dt = {
        let dt_res = world.get_resource::<DeltaTime>().unwrap();
        dt_res.dt as f32
    };

    let mut query = world.query::<(&MovingObject, &mut Transform)>();
    for (_, (moving, transform)) in query.iter_mut() {
        transform.position.x += moving.velocity.x * dt;
        transform.position.y += moving.velocity.y * dt;
        transform.position.z += moving.velocity.z * dt;
    }
}
