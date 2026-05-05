use crate::builders::configs::buffer_configs::UIInstanceBuffer;
use crate::rendering::commands::CreateBuffersCommand;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub fn create_ui_instance_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<UIInstanceBuffer>::new());
}
