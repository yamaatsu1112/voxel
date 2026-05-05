use voxel_engine::*;

use crate::components::{CollisionState, Velocity};

pub fn build_collision_input_data(
    transform: &Transform,
    collider: &Collider,
    velocity: &Velocity,
    dt: f32,
) -> [f32; 9] {
    [
        transform.position.x,
        transform.position.y,
        transform.position.z,
        collider.half_extent.x,
        collider.half_extent.y,
        collider.half_extent.z,
        velocity.velocity.x * dt,
        velocity.velocity.y * dt,
        velocity.velocity.z * dt,
    ]
}

pub fn apply_collision_response_data(
    transform: &mut Transform,
    velocity: &mut Velocity,
    collision_state: &mut CollisionState,
    corrected_pos: [f32; 3],
    collision_flags: [f32; 3],
) {
    transform.position.x = corrected_pos[0];
    transform.position.y = corrected_pos[1];
    transform.position.z = corrected_pos[2];

    if collision_flags[0] > 0.5 {
        velocity.velocity.x = 0.0;
    }
    if collision_flags[1] > 0.5 {
        velocity.velocity.y = 0.0;
    }
    if collision_flags[2] > 0.5 {
        velocity.velocity.z = 0.0;
    }

    collision_state.collided_x = collision_flags[0] > 0.5;
    collision_state.collided_y = collision_flags[1] > 0.5;
    collision_state.collided_z = collision_flags[2] > 0.5;
    collision_state.has_collision_input = false;
}

pub fn apply_previous_collision_system(world: &mut World) {
    let (corrected_pos, collision_flags) = {
        let compute = world.get_resource::<Executor>().unwrap();
        compute.wait_for_compute::<CollisionComputeSubmitGroup>();
        let data: [f32; 6] = compute
            .read_buffer_data::<CollisionResultBuffer, f32, 6>(0)
            .expect("Failed to read collision result buffer");
        ([data[0], data[1], data[2]], [data[3], data[4], data[5]])
    };

    let target_entity = {
        let query = world.query::<&CollisionState>();
        query
            .iter()
            .find(|(_, collision_state)| collision_state.has_collision_input)
            .map(|(entity, _)| entity)
    };

    let Some(entity) = target_entity else {
        return;
    };

    let Some(mut transform) = world.get_component::<Transform>(entity) else {
        return;
    };
    let Some(mut velocity) = world.get_component::<Velocity>(entity) else {
        return;
    };
    let Some(mut collision_state) = world.get_component::<CollisionState>(entity) else {
        return;
    };
    apply_collision_response_data(
        &mut transform,
        &mut velocity,
        &mut collision_state,
        corrected_pos,
        collision_flags,
    );

    if let Some(transform_mut) = world.get_component_mut::<Transform>(entity) {
        *transform_mut = transform;
    }
    if let Some(velocity_mut) = world.get_component_mut::<Velocity>(entity) {
        *velocity_mut = velocity;
    }
    if let Some(collision_state_mut) = world.get_component_mut::<CollisionState>(entity) {
        *collision_state_mut = collision_state;
    }
}

pub fn prepare_collision_input_system(world: &mut World) {
    let target_entity = {
        let query = world.query::<(&Transform, &Collider, &Velocity, &CollisionState)>();
        query.iter().next().map(|(entity, _)| entity)
    };

    let Some(entity) = target_entity else {
        return;
    };

    let Some(transform) = world.get_component::<Transform>(entity) else {
        return;
    };
    let Some(collider) = world.get_component::<Collider>(entity) else {
        return;
    };
    let Some(velocity) = world.get_component::<Velocity>(entity) else {
        return;
    };

    let dt = world.get_resource::<DeltaTime>().unwrap().dt as f32;
    let data = build_collision_input_data(&transform, &collider, &velocity, dt);

    {
        let mut compute = world.get_resource_mut::<Executor>().unwrap();
        compute.upload_to_buffer::<NoMutex, CollisionInputBuffer, f32>(0, &data, 0);
    }

    if let Some(collision_state) = world.get_component_mut::<CollisionState>(entity) {
        collision_state.has_collision_input = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_engine::engine_types::Vec3;

    #[test]
    fn test_build_collision_input_data_writes_position_half_extent_and_displacement() {
        let transform = Transform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
        };
        let collider = Collider {
            half_extent: Vec3::new(0.5, 1.0, 0.25),
        };
        let velocity = Velocity {
            velocity: Vec3::new(-3.0, 4.0, 5.0),
        };

        let data = build_collision_input_data(&transform, &collider, &velocity, 0.5);
        assert_eq!(data, [1.0, 2.0, 3.0, 0.5, 1.0, 0.25, -1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_apply_collision_response_sets_grounded_and_resets_colliding_velocity() {
        let mut transform = Transform::default();
        let mut velocity = Velocity {
            velocity: Vec3::new(2.0, -4.0, 1.0),
        };
        let mut collision_state = CollisionState {
            has_collision_input: true,
            ..Default::default()
        };

        apply_collision_response_data(
            &mut transform,
            &mut velocity,
            &mut collision_state,
            [0.0, 1.2, 0.0],
            [0.0, 1.0, 0.0],
        );

        assert_eq!(transform.position.y, 1.2);
        assert_eq!(velocity.velocity.x, 2.0);
        assert_eq!(velocity.velocity.y, 0.0);
        assert_eq!(velocity.velocity.z, 1.0);
        assert!(collision_state.collided_y);
        assert!(!collision_state.has_collision_input);
    }

    #[test]
    fn test_apply_collision_response_clears_grounded_when_no_y_collision() {
        let mut transform = Transform::default();
        let mut velocity = Velocity {
            velocity: Vec3::new(2.0, -4.0, 1.0),
        };
        let mut collision_state = CollisionState {
            collided_y: true,
            has_collision_input: true,
            ..Default::default()
        };

        apply_collision_response_data(
            &mut transform,
            &mut velocity,
            &mut collision_state,
            [2.0, 2.95, 4.0],
            [0.0, 0.0, 0.0],
        );

        assert!(!collision_state.collided_y);
        assert_eq!(velocity.velocity.x, 2.0);
        assert_eq!(velocity.velocity.y, -4.0);
        assert_eq!(velocity.velocity.z, 1.0);
    }

    #[test]
    fn test_apply_collision_response_resets_x_and_z_velocity_on_wall_collision() {
        let mut transform = Transform::default();
        let mut velocity = Velocity {
            velocity: Vec3::new(3.0, 0.0, -2.0),
        };
        let mut collision_state = CollisionState {
            has_collision_input: true,
            ..Default::default()
        };

        apply_collision_response_data(
            &mut transform,
            &mut velocity,
            &mut collision_state,
            [0.5, 0.0, 0.8],
            [1.0, 0.0, 1.0],
        );

        assert_eq!(velocity.velocity.x, 0.0);
        assert_eq!(velocity.velocity.y, 0.0);
        assert_eq!(velocity.velocity.z, 0.0);
        assert!(!collision_state.collided_y);
        assert_eq!(transform.position.x, 0.5);
        assert_eq!(transform.position.z, 0.8);
    }

    #[test]
    fn test_apply_collision_response_no_collision_preserves_velocity() {
        let mut transform = Transform::default();
        let mut velocity = Velocity {
            velocity: Vec3::new(1.0, -2.0, 3.0),
        };
        let mut collision_state = CollisionState {
            has_collision_input: true,
            ..Default::default()
        };

        apply_collision_response_data(
            &mut transform,
            &mut velocity,
            &mut collision_state,
            [1.0, -2.0, 3.0],
            [0.0, 0.0, 0.0],
        );

        assert_eq!(velocity.velocity.x, 1.0);
        assert_eq!(velocity.velocity.y, -2.0);
        assert_eq!(velocity.velocity.z, 3.0);
        assert!(!collision_state.collided_y);
        assert!(!collision_state.has_collision_input);
    }
}
