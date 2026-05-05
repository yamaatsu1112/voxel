use crate::vulkan::resource_config::{BufferConfig, Fixed};
use ash::vk;

engine_macro::define_buffer! {
    pub struct CollisionInputBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: std::mem::size_of::<[f32; 9]>() as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::from_raw(
            vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        ),
    };
}

engine_macro::define_buffer! {
    pub struct CollisionResultBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: std::mem::size_of::<[f32; 6]>() as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::from_raw(
            vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        ),
    };
}
