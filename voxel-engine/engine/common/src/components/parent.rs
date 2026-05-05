use crate::ecs::component::Component;
use crate::ecs::entity::Entity;
use engine_macro::Component;

#[derive(Component, Clone, Copy, Debug)]
pub struct Parent {
    pub entity: Entity,
}

impl Parent {
    pub fn new(entity: Entity) -> Self {
        Self { entity }
    }
}
