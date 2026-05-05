use crate::world::World;

pub type System = fn(&mut World);
