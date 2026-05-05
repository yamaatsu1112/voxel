use voxel_engine::engine_types::Vec3;
use voxel_engine::*;

use crate::components::{CameraRef, Player, Velocity};
use crate::utils::{calculate_forward_vector, calculate_right_vector};

pub const MOVE_SPEED: f32 = 1.0;

pub fn movement_system(world: &mut World) {
    let player_data: Option<(Entity, Entity)> = {
        let query = world.query::<(&Player, &CameraRef)>();
        query
            .iter()
            .next()
            .map(|(entity, (_, camera_ref))| (entity, camera_ref.entity))
    };

    let Some((player_entity, camera_entity)) = player_data else {
        return;
    };

    let camera_rotation: Option<f32> = world
        .get_component::<Transform>(camera_entity)
        .map(|t| t.rotation.y);

    let Some(camera_yaw) = camera_rotation else {
        return;
    };

    let camera_transform = Transform {
        position: Vec3::new(0.0, 0.0, 0.0),
        rotation: Vec3::new(0.0, camera_yaw, 0.0),
    };
    let forward = calculate_forward_vector(&camera_transform);
    let right = calculate_right_vector(&camera_transform);

    let input_state = world.get_resource::<InputState>().unwrap();

    let mut direction = Vec3::ZERO;
    if input_state.is_key_pressed(Key::KeyW) {
        direction.x += forward.x;
        direction.z += forward.z;
    }
    if input_state.is_key_pressed(Key::KeyS) {
        direction.x -= forward.x;
        direction.z -= forward.z;
    }
    if input_state.is_key_pressed(Key::KeyA) {
        direction.x -= right.x;
        direction.z -= right.z;
    }
    if input_state.is_key_pressed(Key::KeyD) {
        direction.x += right.x;
        direction.z += right.z;
    }

    direction = direction.normalize_or_zero();

    if let Some(velocity) = world.get_component_mut::<Velocity>(player_entity) {
        velocity.velocity.x = direction.x * MOVE_SPEED;
        velocity.velocity.z = direction.z * MOVE_SPEED;
    }
}
