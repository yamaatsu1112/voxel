use voxel_engine::*;

use crate::components::Velocity;

pub fn velocity_apply_system(world: &mut World) {
    let dt = world.get_resource::<DeltaTime>().unwrap().dt as f32;

    let velocity_values: Vec<(Entity, f32, f32, f32)> = {
        let query = world.query::<&Velocity>();
        query
            .iter()
            .map(|(entity, velocity)| {
                (
                    entity,
                    velocity.velocity.x,
                    velocity.velocity.y,
                    velocity.velocity.z,
                )
            })
            .collect()
    };
    for (entity, velocity_x, velocity_y, velocity_z) in velocity_values {
        if let Some(transform) = world.get_component_mut::<Transform>(entity) {
            transform.position.x += velocity_x * dt;
            transform.position.y += velocity_y * dt;
            transform.position.z += velocity_z * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_engine::engine_types::Vec3;

    fn create_test_world(dt: f64) -> World {
        let mut world = World::new();
        world.insert_resource(DeltaTime { dt });
        world
    }

    #[test]
    fn test_velocity_apply_updates_transform_from_velocity() {
        let mut world = create_test_world(0.5);
        let entity = world.create_entity();
        world.add_bundle(
            entity,
            (
                Transform {
                    position: Vec3::new(1.0, 2.0, 3.0),
                    rotation: Vec3::new(0.0, 0.0, 0.0),
                },
                Velocity {
                    velocity: Vec3::new(2.0, 4.0, -6.0),
                },
            ),
        );

        velocity_apply_system(&mut world);

        let transform = world.get_component::<Transform>(entity).unwrap();
        assert!((transform.position.x - 2.0).abs() < 0.0001);
        assert!((transform.position.y - 4.0).abs() < 0.0001);
        assert!((transform.position.z - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_velocity_apply_updates_vertical_position_without_floor_clamp() {
        let mut world = create_test_world(1.0);
        let entity = world.create_entity();
        world.add_bundle(
            entity,
            (
                Transform {
                    position: Vec3::new(0.0, 0.2, 0.0),
                    rotation: Vec3::new(0.0, 0.0, 0.0),
                },
                Velocity {
                    velocity: Vec3::new(0.0, -2.0, 0.0),
                },
            ),
        );

        velocity_apply_system(&mut world);

        let transform = world.get_component::<Transform>(entity).unwrap();
        let velocity = world.get_component::<Velocity>(entity).unwrap();
        assert!((transform.position.y - (-1.8)).abs() < 0.0001);
        assert_eq!(velocity.velocity.y, -2.0);
    }
}
