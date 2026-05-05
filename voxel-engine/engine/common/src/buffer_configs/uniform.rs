use crate::uniform_buffer_object::UniformBufferObject;
use crate::vulkan::resource_config::{BufferConfig, Fixed};
use ash::vk;

engine_macro::define_buffer! {
    /// Uniform buffer for per-frame camera and transform data
    pub struct UniformBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = 1;
    config = BufferConfig {
        size: std::mem::size_of::<UniformBufferObject>() as u64,
        usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
        properties: vk::MemoryPropertyFlags::from_raw(
            vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        ),
    };
}

engine_macro::define_buffer! {
    /// Uniform buffer for compute pipelines (fixed across frames)
    pub struct ComputeUniformBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: std::mem::size_of::<UniformBufferObject>() as u64,
        usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
        properties: vk::MemoryPropertyFlags::from_raw(
            vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        ),
    };
}
