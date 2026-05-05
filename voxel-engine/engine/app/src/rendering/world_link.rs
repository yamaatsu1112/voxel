use crate::ecs::component::Component;
use crate::ecs::entity::Entity;
use crate::ecs::world::World;

use engine_macro::Component;

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldLinkId {
    pub index: u32,
    pub generation: u32,
}

#[derive(Default)]
pub struct WorldLinkAllocator {
    slots: Vec<WorldLinkSlot>,
    free_indices: Vec<u32>,
}

#[derive(Default)]
struct WorldLinkSlot {
    generation: u32,
    allocated: bool,
}

impl WorldLinkAllocator {
    pub fn allocate(&mut self) -> WorldLinkId {
        if let Some(index) = self.free_indices.pop() {
            let slot = self
                .slots
                .get_mut(index as usize)
                .expect("world link free index out of bounds");
            slot.allocated = true;
            return WorldLinkId {
                index,
                generation: slot.generation,
            };
        }

        let index = self.slots.len() as u32;
        self.slots.push(WorldLinkSlot {
            generation: 0,
            allocated: true,
        });
        WorldLinkId {
            index,
            generation: 0,
        }
    }

    pub fn release(&mut self, world_link_id: WorldLinkId) {
        let Some(slot) = self.slots.get_mut(world_link_id.index as usize) else {
            return;
        };
        if !slot.allocated || slot.generation != world_link_id.generation {
            return;
        }

        slot.allocated = false;
        slot.generation = slot.generation.wrapping_add(1);
        self.free_indices.push(world_link_id.index);
    }
}

pub fn ensure_world_link_id(world: &mut World, entity: Entity) -> WorldLinkId {
    if let Some(world_link_id) = world.get_component::<WorldLinkId>(entity) {
        return world_link_id;
    }

    let world_link_id = {
        let mut allocator = world
            .get_resource_mut::<WorldLinkAllocator>()
            .expect("world link allocator missing");
        allocator.allocate()
    };
    world.add_bundle(entity, (world_link_id,));
    world_link_id
}

#[derive(Default)]
pub struct WorldLinkRegistry {
    entries: Vec<Option<WorldLinkEntry>>,
}

#[derive(Clone, Copy)]
pub struct WorldLinkEntry {
    pub generation: u32,
    pub render_entity: Entity,
    pub last_seen_game_tick: u64,
}

impl WorldLinkRegistry {
    pub fn touch(&mut self, world_link_id: WorldLinkId, tick_id: u64) -> Option<Entity> {
        let entry = self.entries.get_mut(world_link_id.index as usize)?;
        let entry = entry.as_mut()?;
        if entry.generation != world_link_id.generation {
            return None;
        }
        entry.last_seen_game_tick = tick_id;
        Some(entry.render_entity)
    }

    pub fn insert(&mut self, world_link_id: WorldLinkId, render_entity: Entity, tick_id: u64) {
        let index = world_link_id.index as usize;
        if self.entries.len() <= index {
            self.entries.resize(index + 1, None);
        }
        self.entries[index] = Some(WorldLinkEntry {
            generation: world_link_id.generation,
            render_entity,
            last_seen_game_tick: tick_id,
        });
    }

    pub fn stale_entries(&self, tick_id: u64) -> Vec<(WorldLinkId, Entity)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let entry = entry.as_ref()?;
                if entry.last_seen_game_tick == tick_id {
                    return None;
                }
                Some((
                    WorldLinkId {
                        index: index as u32,
                        generation: entry.generation,
                    },
                    entry.render_entity,
                ))
            })
            .collect()
    }

    pub fn remove(&mut self, world_link_id: WorldLinkId) {
        let Some(entry) = self.entries.get_mut(world_link_id.index as usize) else {
            return;
        };
        let Some(existing) = entry else {
            return;
        };
        if existing.generation == world_link_id.generation {
            *entry = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WorldLinkAllocator, ensure_world_link_id};
    use crate::ecs::world::World;

    #[test]
    fn allocator_reuses_index_with_incremented_generation() {
        let mut allocator = WorldLinkAllocator::default();

        let first = allocator.allocate();
        allocator.release(first);
        let second = allocator.allocate();

        assert_eq!(second.index, first.index);
        assert_eq!(second.generation, first.generation + 1);
    }

    #[test]
    fn ensure_world_link_id_reuses_existing_component() {
        let mut world = World::new();
        let entity = world.create_entity();
        world.insert_resource(WorldLinkAllocator::default());

        let first = ensure_world_link_id(&mut world, entity);
        let second = ensure_world_link_id(&mut world, entity);

        assert_eq!(first, second);
    }
}
