use crate::buffer_configs::{ComputeUniformBuffer, UniformBuffer};
use crate::rendering::commands::CreateBuffersCommand;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub fn create_uniform_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<UniformBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<ComputeUniformBuffer>::new());
}
