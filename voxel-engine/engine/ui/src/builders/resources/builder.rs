use crate::builders::resources::buffers::create_ui_instance_buffers;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct ResourceBuilder;

impl ResourceBuilder {
    pub fn build_renderer_resources(renderer: &mut Renderer<VulkanRenderer>) {
        create_ui_instance_buffers(renderer);
    }
}
