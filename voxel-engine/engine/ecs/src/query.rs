use crate::archetype::{Archetype, ArchetypeId};
use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

use engine_macro::impl_query_item_tuple;

use std::any::TypeId;
use std::marker::PhantomData;

/// Query for accessing components
pub struct Query<Q> {
    world: *mut World,
    _phantom: PhantomData<Q>,
}

impl<Q> Query<Q> {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            world: world as *mut World,
            _phantom: PhantomData,
        }
    }

    unsafe fn world(&self) -> &World {
        unsafe { &*self.world }
    }
    unsafe fn world_mut(&mut self) -> &mut World {
        unsafe { &mut *self.world }
    }
}

pub trait QueryItem {
    type Item<'w>;
    type ItemMut<'w>;

    fn fetch<'w>(archetype: &'w Archetype, index: usize) -> Option<Self::Item<'w>>;
    fn fetch_mut<'w>(archetype: &'w mut Archetype, index: usize) -> Option<Self::ItemMut<'w>>;
    fn archetypes(world: &World) -> Vec<ArchetypeId>;
}

impl<T: Component + 'static> QueryItem for &T {
    type Item<'w> = &'w T;
    type ItemMut<'w> = &'w T;

    fn fetch<'w>(archetype: &'w Archetype, index: usize) -> Option<Self::Item<'w>> {
        archetype.get_component_at::<T>(index)
    }
    fn fetch_mut<'w>(archetype: &'w mut Archetype, index: usize) -> Option<Self::ItemMut<'w>> {
        archetype.get_component_at::<T>(index)
    }
    fn archetypes(world: &World) -> Vec<ArchetypeId> {
        world.archetypes_by_type_ids(&[TypeId::of::<T>()])
    }
}

impl<T: Component + 'static> QueryItem for &mut T {
    type Item<'w> = &'w T;
    type ItemMut<'w> = &'w mut T;

    fn fetch<'w>(archetype: &'w Archetype, index: usize) -> Option<Self::Item<'w>> {
        archetype.get_component_at::<T>(index)
    }
    fn fetch_mut<'w>(archetype: &'w mut Archetype, index: usize) -> Option<Self::ItemMut<'w>> {
        archetype.get_component_mut_at::<T>(index)
    }
    fn archetypes(world: &World) -> Vec<ArchetypeId> {
        world.archetypes_by_type_ids(&[TypeId::of::<T>()])
    }
}

impl_query_item_tuple!((&T1, &T2));
impl_query_item_tuple!((&T1, &mut T2));
impl_query_item_tuple!((&mut T1, &T2));
impl_query_item_tuple!((&mut T1, &mut T2));
impl_query_item_tuple!((&T1, &T2, &T3));
impl_query_item_tuple!((&T1, &T2, &mut T3));
impl_query_item_tuple!((&T1, &mut T2, &T3));
impl_query_item_tuple!((&T1, &mut T2, &mut T3));
impl_query_item_tuple!((&mut T1, &T2, &T3));
impl_query_item_tuple!((&mut T1, &T2, &mut T3));
impl_query_item_tuple!((&mut T1, &mut T2, &T3));
impl_query_item_tuple!((&mut T1, &mut T2, &mut T3));

// 4-element tuples
impl_query_item_tuple!((&T1, &T2, &T3, &T4));
impl_query_item_tuple!((&T1, &T2, &T3, &mut T4));
impl_query_item_tuple!((&T1, &T2, &mut T3, &T4));
impl_query_item_tuple!((&T1, &T2, &mut T3, &mut T4));
impl_query_item_tuple!((&T1, &mut T2, &T3, &T4));
impl_query_item_tuple!((&T1, &mut T2, &T3, &mut T4));
impl_query_item_tuple!((&T1, &mut T2, &mut T3, &T4));
impl_query_item_tuple!((&T1, &mut T2, &mut T3, &mut T4));
impl_query_item_tuple!((&mut T1, &T2, &T3, &T4));
impl_query_item_tuple!((&mut T1, &T2, &T3, &mut T4));
impl_query_item_tuple!((&mut T1, &T2, &mut T3, &T4));
impl_query_item_tuple!((&mut T1, &T2, &mut T3, &mut T4));
impl_query_item_tuple!((&mut T1, &mut T2, &T3, &T4));
impl_query_item_tuple!((&mut T1, &mut T2, &T3, &mut T4));
impl_query_item_tuple!((&mut T1, &mut T2, &mut T3, &T4));
impl_query_item_tuple!((&mut T1, &mut T2, &mut T3, &mut T4));

pub struct QueryIter<'w, Q: QueryItem> {
    // Iterator over remaining archetypes to process
    archetype_ids: Vec<ArchetypeId>,
    // Current archetype being processed (None when all are exhausted)
    current_archetype: usize,
    // Index of the current entity within the current archetype
    current_entity_index: usize,
    // PhantomData to hold the query type Q
    _phantom: PhantomData<&'w Q>,
    world: &'w World,
}

impl<'w, Q: QueryItem> QueryIter<'w, Q> {
    fn new(world: &'w World) -> Self {
        let archetype_ids = Q::archetypes(world);

        Self {
            archetype_ids,
            current_archetype: 0,
            current_entity_index: 0,
            _phantom: PhantomData,
            world,
        }
    }
}

impl<'w, Q: QueryItem> Iterator for QueryIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Get current archetype - return None if all archetypes are exhausted
            if self.current_archetype >= self.archetype_ids.len() {
                return None;
            }
            let archetype_id = self.archetype_ids[self.current_archetype];
            let archetype = self.world.get_archetype(archetype_id)?;

            let entities = archetype.entities();

            // Check if we've exhausted entities in this archetype
            if self.current_entity_index >= entities.len() {
                // Move to next archetype and reset entity index
                self.current_archetype += 1;
                self.current_entity_index = 0;
                continue;
            }

            // Get entity index and fetch the query item
            // The entity_idx is used to access components in the ComponentData arrays
            if !archetype.is_deleted_at(self.current_entity_index) {
                if let Some(item) = Q::fetch(archetype, self.current_entity_index) {
                    let val = Some((entities[self.current_entity_index], item));
                    self.current_entity_index += 1;
                    return val;
                }
            } else {
                self.current_entity_index += 1;
            }
        }
    }
}

pub struct QueryIterMut<'w, Q: QueryItem> {
    // Iterator over remaining archetypes to process
    archetype_ids: Vec<ArchetypeId>,
    // Raw pointer to the current archetype (None when all are exhausted)
    // Raw pointer keeps borrow scope short while still letting us get &mut access.
    current_archetype: usize,
    // Index of the current entity within the current archetype
    current_entity_index: usize,
    // PhantomData to hold the query type Q
    _phantom: PhantomData<&'w Q>,
    world: *mut World,
}

impl<'w, Q: QueryItem> QueryIterMut<'w, Q> {
    fn new(world: &'w mut World) -> Self {
        let archetype_ids = Q::archetypes(world);
        Self {
            archetype_ids,
            current_archetype: 0,
            current_entity_index: 0,
            _phantom: PhantomData,
            world,
        }
    }
}

impl<'w, Q: QueryItem> Iterator for QueryIterMut<'w, Q> {
    type Item = (Entity, Q::ItemMut<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Get current archetype - return None if all archetypes are exhausted
            if self.current_archetype >= self.archetype_ids.len() {
                return None;
            }
            let archetype_id = self.archetype_ids[self.current_archetype];

            let world = unsafe { &mut *self.world }; // TODO: Don't use unsafe
            let archetype = world.get_archetype_mut(archetype_id)?;

            let entities = archetype.entities();

            // Check if we've exhausted entities in this archetype
            if self.current_entity_index >= entities.len() {
                // Move to next archetype and reset entity index
                self.current_archetype += 1;
                self.current_entity_index = 0;
                continue;
            }

            let entity = entities[self.current_entity_index];

            // Get entity index and fetch the query item
            // The entity_idx is used to access components in the ComponentData arrays
            if !archetype.is_deleted_at(self.current_entity_index) {
                if let Some(item) = Q::fetch_mut(archetype, self.current_entity_index) {
                    let val = Some((entity, item));
                    self.current_entity_index += 1;
                    return val;
                } else {
                    panic!("QueryIterMut::next: Should not be reach here.");
                }
            } else {
                self.current_entity_index += 1;
            }
        }
    }
}

pub trait QueryExt<Q: QueryItem> {
    /// Create an iterator over query results
    fn iter<'w>(&'w self) -> QueryIter<'w, Q>;
    fn iter_mut<'w>(&'w mut self) -> QueryIterMut<'w, Q>;
}

impl<Q: QueryItem> QueryExt<Q> for Query<Q> {
    fn iter<'w>(&'w self) -> QueryIter<'w, Q> {
        unsafe { QueryIter::new(self.world()) }
    }
    fn iter_mut<'w>(&'w mut self) -> QueryIterMut<'w, Q> {
        unsafe { QueryIterMut::new(self.world_mut()) }
    }
}
