use crate::archetype::Archetype;
use crate::component::Component;
use std::any::TypeId;

/// Trait for types that can be inserted as a group of components into an entity.
///
/// This trait is implemented for tuples of components, allowing multiple components
/// to be added to an entity in a single operation.
pub trait Bundle {
    /// Returns the TypeIds of all components in this bundle
    fn get_type_ids() -> Vec<TypeId>;

    /// Returns the sizes of all components in this bundle
    fn get_sizes() -> Vec<usize>;

    /// Add all component types to the archetype
    fn add_to_archetype(archetype: &mut Archetype);

    /// Set all component values at the given index
    fn set_at(self, archetype: &mut Archetype, index: usize);
}

// Implement Bundle for single component (base case)
impl<T: Component + 'static> Bundle for (T,) {
    fn get_type_ids() -> Vec<TypeId> {
        vec![TypeId::of::<T>()]
    }

    fn get_sizes() -> Vec<usize> {
        vec![std::mem::size_of::<T>()]
    }

    fn add_to_archetype(archetype: &mut Archetype) {
        archetype.add_component::<T>();
    }

    fn set_at(self, archetype: &mut Archetype, index: usize) {
        archetype.set_component_at::<T>(index, self.0);
    }
}

// Implement Bundle for tuples of 2 to 12 components
macro_rules! impl_bundle {
    ($($T:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($T: Component + 'static),+> Bundle for ($($T,)+) {
            fn get_type_ids() -> Vec<TypeId> {
                vec![$(TypeId::of::<$T>()),+]
            }

            fn get_sizes() -> Vec<usize> {
                vec![$(std::mem::size_of::<$T>()),+]
            }

            fn add_to_archetype(archetype: &mut Archetype) {
                $(archetype.add_component::<$T>();)+
            }

            fn set_at(self, archetype: &mut Archetype, index: usize) {
                let ($($T,)+) = self;
                $(archetype.set_component_at::<$T>(index, $T);)+
            }
        }
    };
}

impl_bundle!(T0, T1);
impl_bundle!(T0, T1, T2);
impl_bundle!(T0, T1, T2, T3);
impl_bundle!(T0, T1, T2, T3, T4);
impl_bundle!(T0, T1, T2, T3, T4, T5);
impl_bundle!(T0, T1, T2, T3, T4, T5, T6);
impl_bundle!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_bundle!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
impl_bundle!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_bundle!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_bundle!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::query::QueryExt;
    use crate::world::World;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
        z: f32,
    }
    impl Component for Position {}

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Velocity {
        x: f32,
        y: f32,
        z: f32,
    }
    impl Component for Velocity {}

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Health {
        value: f32,
    }
    impl Component for Health {}

    #[test]
    fn test_bundle_type_ids() {
        let type_ids = <(Position, Velocity)>::get_type_ids();
        assert_eq!(type_ids.len(), 2);
        assert!(type_ids.contains(&TypeId::of::<Position>()));
        assert!(type_ids.contains(&TypeId::of::<Velocity>()));
    }

    #[test]
    fn test_bundle_sizes() {
        let sizes = <(Position, Velocity)>::get_sizes();
        assert_eq!(sizes.len(), 2);
        assert_eq!(sizes[0], std::mem::size_of::<Position>());
        assert_eq!(sizes[1], std::mem::size_of::<Velocity>());
    }

    #[test]
    fn test_add_bundle() {
        let mut world = World::new();
        let entity = world.create_entity();

        let pos = Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let vel = Velocity {
            x: 4.0,
            y: 5.0,
            z: 6.0,
        };

        world.add_bundle(entity, (pos, vel));

        // Query to verify components were added
        let query = world.query::<(&Position, &Velocity)>();
        let mut found = false;
        query.iter().for_each(|(_entity, (p, v))| {
            assert_eq!(*p, pos);
            assert_eq!(*v, vel);
            found = true;
        });
        assert!(found, "Entity with bundle components not found");
    }

    #[test]
    fn test_add_bundle_to_existing_entity_with_components() {
        let mut world = World::new();
        let entity = world.create_entity();

        // Add initial component
        let health = Health { value: 100.0 };
        world.add_bundle(entity, (health,));

        // Add bundle
        let pos = Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let vel = Velocity {
            x: 4.0,
            y: 5.0,
            z: 6.0,
        };
        world.add_bundle(entity, (pos, vel));

        // Query to verify all components are present
        let query = world.query::<(&Position, &Velocity, &Health)>();
        let mut found = false;
        query.iter().for_each(|(_entity, (p, v, h))| {
            assert_eq!(*p, pos);
            assert_eq!(*v, vel);
            assert_eq!(*h, health);
            found = true;
        });
        assert!(found, "Entity with all components not found");
    }
}
