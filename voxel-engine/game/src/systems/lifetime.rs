use voxel_engine::*;

use crate::components::Lifetime;

/// System to manage entity lifetimes.
/// Decrements remaining lifetime and deletes entities when expired.
pub fn lifetime_system(world: &mut World) {
    let dt = {
        let dt_res = world.get_resource::<DeltaTime>().unwrap();
        dt_res.dt as f32
    };

    // Collect entities to delete
    let mut to_delete = Vec::new();

    {
        let mut query = world.query::<&mut Lifetime>();
        for (entity, lifetime) in query.iter_mut() {
            lifetime.remaining -= dt;
            if lifetime.remaining <= 0.0 {
                to_delete.push(entity);
            }
        }
    }

    // Delete expired entities
    for entity in to_delete {
        world.delete_entity(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifetime_system_decrements_lifetime() {
        let mut world = World::new();

        // Setup DeltaTime resource
        world.insert_resource(DeltaTime { dt: 0.5 });

        // Create entity with Lifetime
        let entity = world.create_entity();
        world.add_bundle(entity, (Lifetime { remaining: 2.0 },));

        // Run system
        lifetime_system(&mut world);

        // Check that lifetime was decremented
        let query = world.query::<&Lifetime>();
        let lifetimes: Vec<_> = query.iter().collect();
        assert_eq!(lifetimes.len(), 1);
        assert!((lifetimes[0].1.remaining - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lifetime_system_deletes_expired_entity() {
        let mut world = World::new();

        // Setup DeltaTime resource
        world.insert_resource(DeltaTime { dt: 1.0 });

        // Create entity with short lifetime
        let entity = world.create_entity();
        world.add_bundle(entity, (Lifetime { remaining: 0.5 },));

        // Run system - entity should be deleted
        lifetime_system(&mut world);

        // Check that entity was deleted
        let query = world.query::<&Lifetime>();
        let lifetimes: Vec<_> = query.iter().collect();
        assert_eq!(lifetimes.len(), 0);
    }

    #[test]
    fn test_lifetime_system_keeps_entities_with_remaining_time() {
        let mut world = World::new();

        // Setup DeltaTime resource
        world.insert_resource(DeltaTime { dt: 0.1 });

        // Create multiple entities
        let entity1 = world.create_entity();
        world.add_bundle(entity1, (Lifetime { remaining: 2.0 },));

        let entity2 = world.create_entity();
        world.add_bundle(entity2, (Lifetime { remaining: 0.05 },)); // Will expire

        let entity3 = world.create_entity();
        world.add_bundle(entity3, (Lifetime { remaining: 1.0 },));

        // Run system
        lifetime_system(&mut world);

        // Check that only expired entity was deleted
        let query = world.query::<&Lifetime>();
        let lifetimes: Vec<_> = query.iter().collect();
        assert_eq!(lifetimes.len(), 2);
    }
}
