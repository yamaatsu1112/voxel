use crate::components::local_transform::LocalTransform;
use crate::components::parent::Parent;
use crate::components::transform::Transform;
use crate::ecs::query::QueryExt;
use crate::ecs::world::World;

pub fn local_transform_system(world: &mut World) {
    let mut query = world.query::<(&Parent, &LocalTransform, &mut Transform)>();

    for (_entity, (parent, local_transform, transform)) in query.iter_mut() {
        let parent_entity = parent.entity;
        if let Some(parent_transform) = world.get_component::<Transform>(parent_entity) {
            transform.position.x = parent_transform.position.x + local_transform.position.x;
            transform.position.y = parent_transform.position.y + local_transform.position.y;
            transform.position.z = parent_transform.position.z + local_transform.position.z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_app::engine_types::Vec3;

    #[test]
    fn test_local_transform_system_basic() {
        let mut world = World::new();

        // Create parent entity with Transform
        let parent_entity = world.create_entity();
        world.add_bundle(
            parent_entity,
            (Transform {
                position: Vec3::new(10.0, 20.0, 30.0),
                rotation: Vec3::new(0.0, 0.0, 0.0),
            },),
        );

        // Create child entity with Parent, LocalTransform, and Transform
        let child_entity = world.create_entity();
        world.add_bundle(
            child_entity,
            (
                Parent::new(parent_entity),
                LocalTransform::new(Vec3::new(1.0, 2.0, 3.0)),
                Transform::default(),
            ),
        );

        // Run the system
        local_transform_system(&mut world);

        // Check child's world position
        let child_transform = world.get_component::<Transform>(child_entity).unwrap();
        assert!((child_transform.position.x - 11.0).abs() < 0.001);
        assert!((child_transform.position.y - 22.0).abs() < 0.001);
        assert!((child_transform.position.z - 33.0).abs() < 0.001);
    }

    #[test]
    fn test_local_transform_system_child_rotation_preserved() {
        let mut world = World::new();

        let parent_entity = world.create_entity();
        world.add_bundle(
            parent_entity,
            (Transform {
                position: Vec3::new(0.0, 0.0, 0.0),
                rotation: Vec3::new(0.0, 0.0, 0.0),
            },),
        );

        let child_entity = world.create_entity();
        world.add_bundle(
            child_entity,
            (
                Parent::new(parent_entity),
                LocalTransform::new(Vec3::new(0.0, 0.0, 0.0)),
                Transform {
                    position: Vec3::new(999.0, 999.0, 999.0), // Will be overwritten
                    rotation: Vec3::new(45.0, 90.0, 180.0),   // Should be preserved
                },
            ),
        );

        local_transform_system(&mut world);

        let child_transform = world.get_component::<Transform>(child_entity).unwrap();
        // Position should be updated to parent's position
        assert!(child_transform.position.x.abs() < 0.001);
        assert!(child_transform.position.y.abs() < 0.001);
        assert!(child_transform.position.z.abs() < 0.001);
        // Rotation should be preserved
        assert!((child_transform.rotation.x - 45.0).abs() < 0.001);
        assert!((child_transform.rotation.y - 90.0).abs() < 0.001);
        assert!((child_transform.rotation.z - 180.0).abs() < 0.001);
    }

    #[test]
    fn test_local_transform_system_no_parent_transform() {
        let mut world = World::new();

        // Create parent without Transform
        let parent_entity = world.create_entity();

        // Create child with Parent pointing to entity without Transform
        let child_entity = world.create_entity();
        world.add_bundle(
            child_entity,
            (
                Parent::new(parent_entity),
                LocalTransform::new(Vec3::new(1.0, 2.0, 3.0)),
                Transform {
                    position: Vec3::new(100.0, 200.0, 300.0),
                    rotation: Vec3::new(0.0, 0.0, 0.0),
                },
            ),
        );

        // System should not crash, child transform should remain unchanged
        local_transform_system(&mut world);

        let child_transform = world.get_component::<Transform>(child_entity).unwrap();
        assert!((child_transform.position.x - 100.0).abs() < 0.001);
        assert!((child_transform.position.y - 200.0).abs() < 0.001);
        assert!((child_transform.position.z - 300.0).abs() < 0.001);
    }
}
