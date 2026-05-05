mod buffers;

use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub use buffers::create_uniform_buffers;

pub fn setup_common_renderer(renderer: &mut Renderer<VulkanRenderer>) {
    create_uniform_buffers(renderer);
}
