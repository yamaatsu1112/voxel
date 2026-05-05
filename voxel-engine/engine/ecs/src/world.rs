use crate::archetype::Archetype;
use crate::archetype::ArchetypeId;
use crate::bundle::Bundle;
use crate::entity::Entity;
use crate::query::Query;
use crate::resource::{Res, ResMut, Resource};

use std::any::TypeId;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct EntityLocation {
    pub archetype_id: ArchetypeId,
    pub index: usize,
}

pub struct World {
    next_entity_id: u32,
    free_entities: Vec<Entity>,
    entity_to_archetype: HashMap<Entity, EntityLocation>,
    archetypes: HashMap<ArchetypeId, Archetype>,
    type_ids: HashMap<ArchetypeId, Vec<TypeId>>,
    type_id_to_archetype_ids: HashMap<TypeId, Vec<ArchetypeId>>,
    resources: Resource,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        let archetype_id = ArchetypeId::new(&mut vec![]);
        let archetype = Archetype::new();

        let mut archetypes = HashMap::new();
        archetypes.insert(archetype_id, archetype);

        let mut type_ids: HashMap<ArchetypeId, Vec<TypeId>> = HashMap::new();
        type_ids.insert(archetype_id, vec![]);

        Self {
            next_entity_id: 0,
            free_entities: Vec::new(),
            entity_to_archetype: HashMap::new(),
            archetypes,
            type_ids,
            type_id_to_archetype_ids: HashMap::new(),
            resources: Resource::new(),
        }
    }

    pub fn create_entity(&mut self) -> Entity {
        let entity = self.get_free_entity();
        let archetype_id = ArchetypeId::new(&mut vec![]);

        let archetype = self.archetypes.get_mut(&archetype_id).unwrap();
        let index = archetype.add_entity(entity);

        self.entity_to_archetype.insert(
            entity,
            EntityLocation {
                archetype_id,
                index,
            },
        );

        entity
    }

    pub fn delete_entity(&mut self, entity: Entity) {
        if let Some(location) = self.entity_to_archetype.remove(&entity) {
            if let Some(archetype) = self.archetypes.get_mut(&location.archetype_id) {
                archetype.remove_entity_at(location.index);
            }
            // Increment generation on deletion
            let next_entity = Entity::new(entity.id, entity.generation + 1);
            self.free_entities.push(next_entity);
        }
    }

    fn get_free_entity(&mut self) -> Entity {
        if let Some(entity) = self.free_entities.pop() {
            entity // Generation was already incremented on deletion
        } else {
            let entity = Entity::new(self.next_entity_id, 0); // generation=0 for new entities
            self.next_entity_id += 1;
            entity
        }
    }

    pub fn add_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        let Some(location) = self.entity_to_archetype.get(&entity) else {
            println!("The entity does not exist");
            return;
        };
        let old_archetype_id = location.archetype_id;
        let old_index = location.index;

        let Some(mut old_archetype) = self.archetypes.remove(&old_archetype_id) else {
            println!("The archetype does not exist");
            return;
        };

        let Some(old_type_ids) = self.type_ids.get(&old_archetype_id) else {
            println!("The type ids do not exist");
            return;
        };

        // Remove entity from old archetype
        old_archetype.remove_entity_at(old_index);

        // Create new type_ids by merging old and bundle type ids
        let bundle_type_ids = B::get_type_ids();
        let mut new_type_ids = old_type_ids.clone();

        // Add bundle type ids that don't already exist
        for type_id in &bundle_type_ids {
            if !new_type_ids.contains(type_id) {
                new_type_ids.push(*type_id);
            }
        }

        let new_archetype_id = ArchetypeId::new(&mut new_type_ids); // new_type_ids is sorted

        let mut archetype_clone = old_archetype.empty_clone();
        if let Some(new_archetype) = self.archetypes.get_mut(&new_archetype_id) {
            // if new archetype exists
            let index = new_archetype.add_entity(entity);
            old_archetype.copy_components_at(old_index, new_archetype, index, old_type_ids);
            bundle.set_at(new_archetype, index);
            self.entity_to_archetype.insert(
                entity,
                EntityLocation {
                    archetype_id: new_archetype_id,
                    index,
                },
            );
        } else {
            // if new archetype does not exist
            let index: usize = archetype_clone.add_entity(entity);
            old_archetype.copy_components_at(old_index, &mut archetype_clone, index, old_type_ids);
            B::add_to_archetype(&mut archetype_clone);
            bundle.set_at(&mut archetype_clone, index);
            self.archetypes.insert(new_archetype_id, archetype_clone);
            self.entity_to_archetype.insert(
                entity,
                EntityLocation {
                    archetype_id: new_archetype_id,
                    index,
                },
            );
            for type_id in &new_type_ids {
                let archetype_ids = self.type_id_to_archetype_ids.entry(*type_id).or_default();
                match archetype_ids.binary_search(&new_archetype_id) {
                    Ok(_) => {}
                    Err(index) => {
                        archetype_ids.insert(index, new_archetype_id);
                    }
                }
            }
            self.type_ids.insert(new_archetype_id, new_type_ids);
        }
        self.archetypes.insert(old_archetype_id, old_archetype);
    }

    pub fn get_resource<T: 'static>(&self) -> Option<Res<'_, T>> {
        self.resources.get::<T>().map(Res::new)
    }

    pub fn get_resource_mut<T: 'static>(&mut self) -> Option<ResMut<'_, T>> {
        self.resources.get_mut::<T>().map(ResMut::new)
    }

    pub fn insert_resource<T: Send + 'static>(&mut self, data: T) {
        self.resources.insert(data);
    }

    pub fn remove_resource<T: 'static>(&mut self) {
        self.resources.remove::<T>();
    }

    /// Get all archetypes (for internal use by Query)
    pub(crate) fn get_archetype(&self, archetype_id: ArchetypeId) -> Option<&Archetype> {
        self.archetypes.get(&archetype_id)
    }

    /// Get all archetypes mutably (for internal use by Query)
    pub(crate) fn get_archetype_mut(
        &mut self,
        archetype_id: ArchetypeId,
    ) -> Option<&mut Archetype> {
        self.archetypes.get_mut(&archetype_id)
    }

    pub(crate) fn archetypes_by_type_ids(&self, type_ids: &[TypeId]) -> Vec<ArchetypeId> {
        let mut archetype_ids = Vec::new();
        for (i, type_id) in type_ids.iter().enumerate() {
            if let Some(type_archetype_ids) = self.type_id_to_archetype_ids.get(type_id) {
                if i == 0 {
                    archetype_ids = type_archetype_ids.clone();
                } else {
                    archetype_ids = Self::get_intersect(&archetype_ids, type_archetype_ids);
                }
            } else {
                return Vec::new();
            }
        }
        archetype_ids
    }

    pub fn query<Q>(&mut self) -> Query<Q> {
        Query::new(self)
    }

    fn get_intersect<T: PartialEq + PartialOrd + Copy>(vec1: &[T], vec2: &[T]) -> Vec<T> {
        let mut intersect = Vec::new();
        let mut i1 = 0;
        let mut i2 = 0;
        while i1 < vec1.len() && i2 < vec2.len() {
            if vec1[i1] == vec2[i2] {
                intersect.push(vec1[i1]);
                i1 += 1;
                i2 += 1;
            } else if vec1[i1] < vec2[i2] {
                i1 += 1;
            } else {
                i2 += 1;
            }
        }
        intersect
    }

    pub fn get_component<T: crate::component::Component + 'static>(
        &self,
        entity: Entity,
    ) -> Option<T> {
        let location = self.entity_to_archetype.get(&entity)?;
        let archetype = self.archetypes.get(&location.archetype_id)?;
        if archetype.is_deleted_at(location.index) {
            return None;
        }
        archetype.get_component_at::<T>(location.index).copied()
    }

    pub fn get_component_mut<T: crate::component::Component + 'static>(
        &mut self,
        entity: Entity,
    ) -> Option<&mut T> {
        let location = self.entity_to_archetype.get(&entity)?;
        let archetype_id = location.archetype_id;
        let index = location.index;
        let archetype = self.archetypes.get_mut(&archetype_id)?;
        if archetype.is_deleted_at(index) {
            return None;
        }
        archetype.get_component_mut_at::<T>(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use std::any::TypeId;

    #[derive(Clone, Copy, Default)]
    struct Transform {
        position_x: f32,
        position_y: f32,
        position_z: f32,
    }

    impl Component for Transform {}

    #[derive(Clone, Copy, Default)]
    struct Velocity;

    impl Component for Velocity {}

    #[test]
    fn test_get_component_returns_component_when_exists() {
        let mut world = World::new();
        let entity = world.create_entity();
        let transform = Transform {
            position_x: 1.0,
            position_y: 2.0,
            position_z: 3.0,
        };
        world.add_bundle(entity, (transform,));

        let result = world.get_component::<Transform>(entity);
        assert!(result.is_some());
        let t = result.unwrap();
        assert_eq!(t.position_x, 1.0);
        assert_eq!(t.position_y, 2.0);
        assert_eq!(t.position_z, 3.0);
    }

    #[test]
    fn test_get_component_returns_none_when_entity_not_exists() {
        let world = World::new();
        let non_existent_entity = Entity::new(999, 0);

        let result = world.get_component::<Transform>(non_existent_entity);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_component_returns_none_when_component_not_exists() {
        let mut world = World::new();
        let entity = world.create_entity();

        let result = world.get_component::<Transform>(entity);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_component_returns_none_for_deleted_entity() {
        let mut world = World::new();
        let entity = world.create_entity();
        let transform = Transform::default();
        world.add_bundle(entity, (transform,));
        world.delete_entity(entity);

        let result = world.get_component::<Transform>(entity);
        assert!(result.is_none());
    }

    #[test]
    fn test_entity_generation_starts_at_zero() {
        let mut world = World::new();
        let entity = world.create_entity();
        assert_eq!(entity.generation, 0);
    }

    #[test]
    fn test_entity_generation_increments_on_reuse() {
        let mut world = World::new();
        let entity1 = world.create_entity();
        assert_eq!(entity1.id, 0);
        assert_eq!(entity1.generation, 0);

        world.delete_entity(entity1);
        let entity2 = world.create_entity();

        assert_eq!(entity2.id, 0); // Same ID
        assert_eq!(entity2.generation, 1); // Generation incremented
    }

    #[test]
    fn test_archetypes_by_type_ids_returns_empty_when_any_type_is_missing() {
        let mut world = World::new();
        let entity = world.create_entity();
        world.add_bundle(entity, (Transform::default(),));

        let archetype_ids =
            world.archetypes_by_type_ids(&[TypeId::of::<Transform>(), TypeId::of::<Velocity>()]);

        assert!(archetype_ids.is_empty());
    }
}
