use crate::vertex::{FULLSCREEN_VERTICES, INDICES};
use crate::vulkan::resource_config::{BufferConfig, Fixed};
use ash::vk;

engine_macro::define_buffer! {
    /// Fullscreen quad vertex buffer
    pub struct FullscreenVertexBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: std::mem::size_of_val(FULLSCREEN_VERTICES) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Index buffer for rendering
    pub struct IndexBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: std::mem::size_of_val(INDICES) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}
