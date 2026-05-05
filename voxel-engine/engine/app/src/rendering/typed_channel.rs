use crate::ecs::world::World;
use crate::rendering::double_buffer::DoubleBuffer;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Received<T>(T);

impl<T> AsRef<T> for Received<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

pub struct RenderChannel {
    double_buffer: Arc<DoubleBuffer<ChannelData>>,
    has_new_data: Arc<AtomicBool>,
}

impl Clone for RenderChannel {
    fn clone(&self) -> Self {
        Self {
            double_buffer: Arc::clone(&self.double_buffer),
            has_new_data: Arc::clone(&self.has_new_data),
        }
    }
}

impl Default for RenderChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderChannel {
    pub fn new() -> Self {
        Self {
            double_buffer: Arc::new(DoubleBuffer::new(ChannelData::new(), ChannelData::new())),
            has_new_data: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn write<T: Send + 'static, F>(&self, update: F)
    where
        F: FnOnce(&mut Option<T>),
    {
        let mut write_buffer = self.double_buffer.write_buffer();
        write_buffer.update(update);
        self.has_new_data.store(true, Ordering::Release);
    }

    pub fn swap(&self) {
        if !self.has_new_data.swap(false, Ordering::AcqRel) {
            return;
        }

        let swapped_read = self.double_buffer.swap_and_read();
        drop(swapped_read);

        let mut write_buffer = self.double_buffer.write_buffer();
        write_buffer.clear_values();
    }

    pub fn inject_received_resources(&self, world: &mut World) {
        let mut read_buffer = self.double_buffer.read_buffer();
        read_buffer.inject_received(world);
    }
}

pub struct GameChannel {
    double_buffer: Arc<DoubleBuffer<ChannelData>>,
    has_new_data: Arc<AtomicBool>,
}

impl Clone for GameChannel {
    fn clone(&self) -> Self {
        Self {
            double_buffer: Arc::clone(&self.double_buffer),
            has_new_data: Arc::clone(&self.has_new_data),
        }
    }
}

impl Default for GameChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl GameChannel {
    pub fn new() -> Self {
        Self {
            double_buffer: Arc::new(DoubleBuffer::new(ChannelData::new(), ChannelData::new())),
            has_new_data: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn write<T: Send + 'static, F>(&self, update: F)
    where
        F: FnOnce(&mut Option<T>),
    {
        let mut write_buffer = self.double_buffer.write_buffer();
        write_buffer.update(update);
        self.has_new_data.store(true, Ordering::Release);
    }

    pub fn swap(&self) {
        if !self.has_new_data.swap(false, Ordering::AcqRel) {
            return;
        }

        let swapped_read = self.double_buffer.swap_and_read();
        drop(swapped_read);

        let mut write_buffer = self.double_buffer.write_buffer();
        write_buffer.clear_values();
    }

    pub fn inject_received_resources(&self, world: &mut World) {
        let mut read_buffer = self.double_buffer.read_buffer();
        read_buffer.inject_received(world);
    }
}

struct ChannelData {
    slots: HashMap<TypeId, Box<dyn ChannelSlot>>,
}

impl ChannelData {
    fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    fn update<T: Send + 'static, F>(&mut self, update: F)
    where
        F: FnOnce(&mut Option<T>),
    {
        let slot = self
            .slots
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(TypedSlot::<T>::new()));
        let typed_slot = slot
            .as_any_mut()
            .downcast_mut::<TypedSlot<T>>()
            .expect("type mismatch in channel slot");
        update(&mut typed_slot.value);
    }

    fn inject_received(&mut self, world: &mut World) {
        for slot in self.slots.values_mut() {
            slot.inject_received(world);
        }
    }

    fn clear_values(&mut self) {
        for slot in self.slots.values_mut() {
            slot.clear_values();
        }
    }
}

trait ChannelSlot: Any + Send {
    fn inject_received(&mut self, world: &mut World);
    fn clear_values(&mut self);
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct TypedSlot<T> {
    value: Option<T>,
}

impl<T> TypedSlot<T> {
    fn new() -> Self {
        Self { value: None }
    }
}

impl<T: Send + 'static> ChannelSlot for TypedSlot<T> {
    fn inject_received(&mut self, world: &mut World) {
        let Some(value) = self.value.take() else {
            return;
        };
        world.insert_resource(Received(value));
    }

    fn clear_values(&mut self) {
        self.value = None;
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{Received, RenderChannel};
    use crate::ecs::world::World;

    #[test]
    fn send_and_receive_multiple_values_for_same_type() {
        let channel = RenderChannel::new();
        let mut world = World::new();

        channel.write::<u32, _>(|value| *value = Some(1));
        channel.write::<u32, _>(|value| *value = Some(2));
        channel.swap();
        channel.inject_received_resources(&mut world);

        let value = world
            .get_resource::<Received<u32>>()
            .expect("received values missing")
            .as_ref()
            .to_owned();
        assert_eq!(value, 2);
    }

    #[test]
    fn send_and_receive_multiple_types() {
        let channel = RenderChannel::new();
        let mut world = World::new();

        channel.write::<u32, _>(|value| *value = Some(7));
        channel.write::<i32, _>(|value| *value = Some(-3));
        channel.swap();
        channel.inject_received_resources(&mut world);

        let u32_value = world
            .get_resource::<Received<u32>>()
            .expect("u32 missing")
            .as_ref()
            .to_owned();
        let i32_value = world
            .get_resource::<Received<i32>>()
            .expect("i32 missing")
            .as_ref()
            .to_owned();

        assert_eq!(u32_value, 7);
        assert_eq!(i32_value, -3);
    }

    #[test]
    fn swap_clears_next_write_buffer_to_prevent_stale_data_reinsertion() {
        let channel = RenderChannel::new();
        let mut world = World::new();

        channel.write::<u32, _>(|value| *value = Some(10));
        channel.swap();
        channel.inject_received_resources(&mut world);
        let first = world
            .get_resource::<Received<u32>>()
            .expect("first receive missing")
            .as_ref()
            .to_owned();
        assert_eq!(first, 10);

        channel.write::<u32, _>(|value| *value = Some(20));
        channel.swap();
        channel.inject_received_resources(&mut world);
        let second = world
            .get_resource::<Received<u32>>()
            .expect("second receive missing")
            .as_ref()
            .to_owned();
        assert_eq!(second, 20);

        channel.swap();
        channel.inject_received_resources(&mut world);
        let third = world
            .get_resource::<Received<u32>>()
            .expect("third receive missing")
            .as_ref()
            .to_owned();
        assert_eq!(third, 20);
    }

    #[test]
    fn received_resource_keeps_last_value_when_type_is_not_sent_in_next_frame() {
        let channel = RenderChannel::new();
        let mut world = World::new();

        channel.write::<u32, _>(|value| *value = Some(99));
        channel.swap();
        channel.inject_received_resources(&mut world);
        let first = world
            .get_resource::<Received<u32>>()
            .expect("missing first u32 values")
            .as_ref()
            .to_owned();
        assert_eq!(first, 99);

        channel.write::<i32, _>(|value| *value = Some(-1));
        channel.swap();
        channel.inject_received_resources(&mut world);

        let second_u32 = world
            .get_resource::<Received<u32>>()
            .expect("missing second u32 values")
            .as_ref()
            .to_owned();
        let second_i32 = world
            .get_resource::<Received<i32>>()
            .expect("missing i32 values")
            .as_ref()
            .to_owned();

        assert_eq!(second_u32, 99);
        assert_eq!(second_i32, -1);
    }
}
