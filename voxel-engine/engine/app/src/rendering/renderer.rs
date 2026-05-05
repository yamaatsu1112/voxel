use std::sync::Arc;

use crate::rendering::double_buffer::DoubleBuffer;
use crate::rendering::renderer_command::{
    CommandType, FrameCommandQueue, FrameCommands, RendererBackend, TypedRendererCommand,
};
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct Renderer<R: RendererBackend = VulkanRenderer> {
    frame_command_buffer: Arc<DoubleBuffer<FrameCommandQueue<R>>>,
    current_frame_commands: FrameCommands<R>,
}

impl<R: RendererBackend> Renderer<R> {
    pub fn new(frame_command_buffer: Arc<DoubleBuffer<FrameCommandQueue<R>>>) -> Self {
        Self {
            frame_command_buffer,
            current_frame_commands: FrameCommands::default(),
        }
    }

    pub fn add_command<T: TypedRendererCommand<R> + 'static>(&mut self, cmd: T) {
        <T::CmdType as CommandType<R>>::add_to(Box::new(cmd), &mut self.current_frame_commands);
    }

    pub fn submit_frame(&mut self) {
        let mut write_buffer = self.frame_command_buffer.write_buffer();
        let frame_commands = std::mem::take(&mut self.current_frame_commands);
        write_buffer.push(frame_commands);
    }
}

impl Renderer<VulkanRenderer> {}
