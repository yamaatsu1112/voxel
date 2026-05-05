use crate::rendering::renderer_command::{
    CommandType, FrameCommandQueue, FrameCommands, TypedRendererCommand,
};
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct RenderWorldRenderer {
    frame_command_queue: FrameCommandQueue<VulkanRenderer>,
    current_frame_commands: FrameCommands<VulkanRenderer>,
}

impl Default for RenderWorldRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderWorldRenderer {
    pub fn new() -> Self {
        Self {
            frame_command_queue: FrameCommandQueue::default(),
            current_frame_commands: FrameCommands::default(),
        }
    }

    pub fn add_command<T: TypedRendererCommand<VulkanRenderer> + 'static>(&mut self, cmd: T) {
        <T::CmdType as CommandType<VulkanRenderer>>::add_to(
            Box::new(cmd),
            &mut self.current_frame_commands,
        );
    }

    pub fn submit_frame(&mut self) {
        let frame_commands = std::mem::take(&mut self.current_frame_commands);
        self.frame_command_queue.push(frame_commands);
    }

    pub fn frame_command_queue_mut(&mut self) -> &mut FrameCommandQueue<VulkanRenderer> {
        &mut self.frame_command_queue
    }
}
