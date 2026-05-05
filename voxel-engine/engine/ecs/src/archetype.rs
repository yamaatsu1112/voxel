use crate::component::{Component, ComponentStorage};
use crate::entity::Entity;

use std::any::TypeId;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Unique identifier for an archetype based on its component types
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ArchetypeId {
    id: u64,
}

impl ArchetypeId {
    /// Creates an ArchetypeId from a sorted vector of TypeIds
    pub fn new(type_ids: &mut Vec<TypeId>) -> Self {
        // Sort TypeIds to ensure consistent ordering
        type_ids.sort();

        // Hash the sorted TypeIds vector
        let mut hasher = DefaultHasher::new();
        type_ids.hash(&mut hasher);

        Self {
            id: hasher.finish(),
        }
    }
}

pub struct Archetype {
    entities: Vec<Entity>,
    free_indices: Vec<usize>,
    is_deleted: Vec<bool>,
    components: HashMap<TypeId, ComponentStorage>,
}

impl Default for Archetype {
    fn default() -> Self {
        Self::new()
    }
}

impl Archetype {
    /// Creates a new Archetype with the given ArchetypeId
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            free_indices: Vec::new(),
            is_deleted: Vec::new(),
            components: HashMap::new(),
        }
    }

    pub fn add_component<T: Component + 'static>(&mut self) {
        self.components
            .insert(TypeId::of::<T>(), ComponentStorage::new::<T>());
    }

    fn get_free_index(&mut self) -> Option<usize> {
        self.free_indices.pop()
    }

    // TODO: Don't use Option. I can know whether there is a free index or not when pop.  But maybe compiler optimizes.
    fn allocate_entity_slot(&mut self, entity: Entity) -> usize {
        let index = self.get_free_index();

        if let Some(index) = index {
            // if there is a free index
            self.entities[index] = entity;
            self.is_deleted[index] = false;
            index
        } else {
            // if there is no free index
            let index = self.entities.len();
            self.entities.push(entity);
            self.is_deleted.push(false);
            index
        }
    }

    /// Add an entity
    pub fn add_entity(&mut self, entity: Entity) -> usize {
        self.allocate_entity_slot(entity)
    }

    pub fn set_component_at<T: Component + 'static>(&mut self, index: usize, component: T) {
        if let Some(component_storage) = self.components.get_mut(&TypeId::of::<T>()) {
            component_storage.set(index, component);
        }
    }

    pub fn remove_entity_at(&mut self, index: usize) {
        if index < self.entities.len() && !self.is_deleted[index] {
            // TODO: Don't check length.
            self.free_indices.push(index);
            self.is_deleted[index] = true;
        }
    }

    pub fn get_component_at<T: Component + 'static>(&self, index: usize) -> Option<&T> {
        if let Some(component_storage) = self.components.get(&TypeId::of::<T>()) {
            component_storage.get::<T>(index)
        } else {
            None
        }
    }

    pub fn get_component_mut_at<T: Component + 'static>(&mut self, index: usize) -> Option<&mut T> {
        if let Some(component_storage) = self.components.get_mut(&TypeId::of::<T>()) {
            component_storage.get_mut::<T>(index)
        } else {
            None
        }
    }

    pub fn is_deleted_at(&self, index: usize) -> bool {
        if index < self.is_deleted.len() {
            self.is_deleted[index]
        } else {
            true
        }
    }

    pub fn copy_components_at(
        &self,
        index: usize,
        dst_archetype: &mut Archetype,
        dst_index: usize,
        type_ids: &Vec<TypeId>,
    ) {
        for type_id in type_ids {
            if let Some(component_storage) = self.components.get(type_id) {
                let component = component_storage.get_copy(index);
                dst_archetype
                    .components
                    .get_mut(type_id)
                    .unwrap()
                    .set_copy(dst_index, component);
            }
        }
        dst_archetype.entities[dst_index] = self.entities[index];
        // dst_archetype.is_deleted[dst_index] = self.is_deleted[index];
    }

    /// Get all entities in this archetype
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    pub fn empty_clone(&self) -> Archetype {
        let mut components = HashMap::new();
        for (type_id, component_storage) in self.components.iter() {
            components.insert(*type_id, component_storage.empty_clone());
        }
        Self {
            entities: Vec::new(),
            free_indices: Vec::new(),
            is_deleted: Vec::new(),
            components,
        }
    }
}
