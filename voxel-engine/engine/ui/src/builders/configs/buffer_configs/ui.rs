use crate::builders::configs::pipeline_configs::UI_INSTANCE_COUNT;
use crate::components::ui::UIInstance;
use crate::vulkan::resource_config::{BufferConfig, Fixed};
use ash::vk;

engine_macro::define_buffer! {
    /// Per-frame UI instance buffer (device-local)
    pub struct UIInstanceBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = 1;
    config = BufferConfig {
        size: (std::mem::size_of::<UIInstance>() * UI_INSTANCE_COUNT) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}
