use crate::rendering::renderer_command::{OneShot, RendererCommand, TypedRendererCommand};
use crate::vulkan::vulkan_renderer::VulkanRenderer;

#[derive(Clone)]
pub struct RegisterSwapchainResizeCommand {
    command: Box<dyn RendererCommand<VulkanRenderer>>,
}

impl RegisterSwapchainResizeCommand {
    pub fn new(command: Box<dyn RendererCommand<VulkanRenderer>>) -> Self {
        Self { command }
    }
}

impl RendererCommand<VulkanRenderer> for RegisterSwapchainResizeCommand {
    fn execute(&self, renderer: &mut VulkanRenderer) {
        renderer.register_on_swapchain_resize_per_frame(self.command.clone_box());
    }

    fn clone_box(&self) -> Box<dyn RendererCommand<VulkanRenderer>> {
        Box::new(Self {
            command: self.command.clone_box(),
        })
    }
}

impl TypedRendererCommand<VulkanRenderer> for RegisterSwapchainResizeCommand {
    type CmdType = OneShot;
}
