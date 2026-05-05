use std::collections::VecDeque;

use ash::vk;

use crate::rendering::GpuMutex;
use crate::vulkan::resource_config::{
    BufferMarker, ComputePipelineMarker, DescriptorPoolMarker, DescriptorSetMarker,
    GraphicsPipelineMarker, ImageMarker, ImageViewMarker, SamplerMarker,
};
use crate::vulkan::resource_lifetime::{PerFrame as PerFrameLifetime, Persistent};

pub trait RendererBackend {
    fn current_frame_index(&self) -> usize {
        panic!("current_frame_index is not implemented")
    }

    fn create_descriptor_pool<P: DescriptorPoolMarker + Send + Sync + 'static>(
        &mut self,
        _pool_sizes: &[vk::DescriptorPoolSize],
        _max_sets: u32,
        _flags: vk::DescriptorPoolCreateFlags,
    ) {
        panic!("create_descriptor_pool is not implemented")
    }

    fn create_buffers<T: BufferMarker + Send + Sync + 'static>(&mut self) {
        panic!("create_buffers is not implemented")
    }

    fn create_images<T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static>(
        &mut self,
        _size: Option<(u32, u32)>,
    ) {
        panic!("create_images is not implemented")
    }

    fn create_image_views<T: ImageViewMarker<Lifetime = Persistent> + Send + Sync + 'static>(
        &mut self,
    ) {
        panic!("create_image_views is not implemented")
    }

    fn create_sampler<T: SamplerMarker + Send + Sync + 'static>(&mut self) {
        panic!("create_sampler is not implemented")
    }

    fn create_descriptor_sets<T: DescriptorSetMarker + Send + Sync + 'static>(&mut self) {
        panic!("create_descriptor_sets is not implemented")
    }

    fn create_graphics_pipeline<T: GraphicsPipelineMarker + Send + Sync + 'static>(&mut self) {
        panic!("create_graphics_pipeline is not implemented")
    }

    fn create_compute_pipeline<T: ComputePipelineMarker + Send + Sync + 'static>(&mut self) {
        panic!("create_compute_pipeline is not implemented")
    }

    fn bind_buffer<
        D: DescriptorSetMarker<Lifetime = Persistent> + Send + Sync + 'static,
        B: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
    >(
        &mut self,
        _binding: u32,
        _array_index: u32,
        _buffer_index: usize,
        _count: usize,
        _descriptor_type: vk::DescriptorType,
    ) {
        panic!("bind_buffer is not implemented")
    }

    fn bind_per_frame_buffer<
        D: DescriptorSetMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
        B: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        _binding: u32,
        _array_index: u32,
        _buffer_index: usize,
        _count: usize,
        _descriptor_type: vk::DescriptorType,
    ) {
        panic!("bind_per_frame_buffer is not implemented")
    }

    fn bind_image<
        D: DescriptorSetMarker<Lifetime = Persistent> + Send + Sync + 'static,
        V: ImageViewMarker<Lifetime = Persistent> + Send + Sync + 'static,
        S: SamplerMarker + Send + Sync + 'static,
    >(
        &mut self,
        _binding: u32,
        _array_index: u32,
        _image_index: usize,
        _count: usize,
    ) {
        panic!("bind_image is not implemented")
    }

    fn upload_to_buffer<
        M: GpuMutex + 'static,
        T: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
        D: Clone + Send + Sync + 'static,
    >(
        &mut self,
        _index: usize,
        _data: &[D],
        _dst_byte_offset: u64,
    ) {
        panic!("upload_to_buffer is not implemented")
    }

    fn upload_to_per_frame_buffer<
        M: GpuMutex + 'static,
        T: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
        D: Clone + Send + Sync + 'static,
    >(
        &mut self,
        _index: usize,
        _data: &[D],
    ) {
        panic!("upload_to_per_frame_buffer is not implemented")
    }

    fn create_per_frame_images<
        T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        _size: Option<(u32, u32)>,
        _frame_index: usize,
    ) {
        panic!("create_per_frame_images is not implemented")
    }

    fn create_per_frame_image_views<
        T: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        _frame_index: usize,
    ) {
        panic!("create_per_frame_image_views is not implemented")
    }

    fn bind_per_frame_image<
        D: DescriptorSetMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
        V: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
        S: SamplerMarker + Send + Sync + 'static,
    >(
        &mut self,
        _binding: u32,
        _array_index: u32,
        _image_index: usize,
        _count: usize,
    ) {
        panic!("bind_per_frame_image is not implemented")
    }

    fn destroy_per_frame_images<
        T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        _frame_index: usize,
    ) {
        panic!("destroy_per_frame_images is not implemented")
    }

    fn destroy_per_frame_image_views<
        T: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        _frame_index: usize,
    ) {
        panic!("destroy_per_frame_image_views is not implemented")
    }

    fn transition_image_layout<
        M: GpuMutex + 'static,
        T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    >(
        &mut self,
        _index: usize,
        _old_layout: vk::ImageLayout,
        _new_layout: vk::ImageLayout,
        _aspect_mask: vk::ImageAspectFlags,
    ) {
        panic!("transition_image_layout is not implemented")
    }

    fn transition_per_frame_image_layout<
        M: GpuMutex + 'static,
        T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        _frame_index: usize,
        _image_index: usize,
        _old_layout: vk::ImageLayout,
        _new_layout: vk::ImageLayout,
        _aspect_mask: vk::ImageAspectFlags,
    ) {
        panic!("transition_per_frame_image_layout is not implemented")
    }

    fn upload_to_image<
        M: GpuMutex + 'static,
        T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    >(
        &mut self,
        _image_index: usize,
        _data: &[u8],
        _width: u32,
        _height: u32,
        _depth: u32,
    ) {
        panic!("upload_to_image is not implemented")
    }
}

pub trait RendererCommand<R: RendererBackend>: Send + Sync {
    fn clone_box(&self) -> Box<dyn RendererCommand<R>>;
    fn execute(&self, renderer: &mut R);
}

impl<R: RendererBackend> Clone for Box<dyn RendererCommand<R>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[allow(dead_code)]
pub trait CommandType<R: RendererBackend> {
    fn add_to(command: Box<dyn RendererCommand<R>>, frame_commands: &mut FrameCommands<R>);
}

#[allow(dead_code)]
pub struct OneShot;

impl<R: RendererBackend> CommandType<R> for OneShot {
    fn add_to(command: Box<dyn RendererCommand<R>>, frame_commands: &mut FrameCommands<R>) {
        frame_commands.one_shot_commands.push(command);
    }
}

#[allow(dead_code)]
pub struct PerFrame;

impl<R: RendererBackend> CommandType<R> for PerFrame {
    fn add_to(command: Box<dyn RendererCommand<R>>, frame_commands: &mut FrameCommands<R>) {
        frame_commands.per_frame_commands.push(command);
    }
}

#[allow(dead_code)]
pub trait TypedRendererCommand<R: RendererBackend>: RendererCommand<R> {
    type CmdType: CommandType<R>;
}

#[allow(dead_code)]
pub struct FrameCommands<R: RendererBackend> {
    pub per_frame_commands: Vec<Box<dyn RendererCommand<R>>>,
    pub one_shot_commands: Vec<Box<dyn RendererCommand<R>>>,
}

impl<R: RendererBackend> Default for FrameCommands<R> {
    fn default() -> Self {
        Self {
            per_frame_commands: Vec::new(),
            one_shot_commands: Vec::new(),
        }
    }
}

pub struct FrameCommandQueue<R: RendererBackend> {
    queue: VecDeque<FrameCommands<R>>,
}

impl<R: RendererBackend> Default for FrameCommandQueue<R> {
    fn default() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl<R: RendererBackend> FrameCommandQueue<R> {
    pub fn push(&mut self, commands: FrameCommands<R>) {
        self.queue.push_back(commands);
    }

    pub fn pop(&mut self) -> Option<FrameCommands<R>> {
        self.queue.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CommandType, FrameCommandQueue, FrameCommands, OneShot, PerFrame, RendererBackend,
        RendererCommand, TypedRendererCommand,
    };

    #[derive(Default)]
    struct TestBackend;

    impl RendererBackend for TestBackend {}

    #[derive(Clone)]
    struct TestOneShotCommand;

    impl<R: RendererBackend> RendererCommand<R> for TestOneShotCommand {
        fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
            Box::new((*self).clone())
        }

        fn execute(&self, _renderer: &mut R) {}
    }

    impl<R: RendererBackend> TypedRendererCommand<R> for TestOneShotCommand {
        type CmdType = OneShot;
    }

    #[derive(Clone)]
    struct TestPerFrameCommand;

    impl<R: RendererBackend> RendererCommand<R> for TestPerFrameCommand {
        fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
            Box::new((*self).clone())
        }

        fn execute(&self, _renderer: &mut R) {}
    }

    impl<R: RendererBackend> TypedRendererCommand<R> for TestPerFrameCommand {
        type CmdType = PerFrame;
    }

    #[test]
    fn one_shot_command_is_added_to_one_shot_collection() {
        let mut commands = FrameCommands::<TestBackend>::default();
        OneShot::add_to(Box::new(TestOneShotCommand), &mut commands);

        assert_eq!(commands.one_shot_commands.len(), 1);
        assert_eq!(commands.per_frame_commands.len(), 0);
    }

    #[test]
    fn per_frame_command_is_added_to_per_frame_collection() {
        let mut commands = FrameCommands::<TestBackend>::default();
        PerFrame::add_to(Box::new(TestPerFrameCommand), &mut commands);

        assert_eq!(commands.one_shot_commands.len(), 0);
        assert_eq!(commands.per_frame_commands.len(), 1);
    }

    #[test]
    fn boxed_renderer_command_can_be_cloned() {
        let command: Box<dyn RendererCommand<TestBackend>> = Box::new(TestOneShotCommand);

        let cloned = command.clone();
        let mut commands = FrameCommands::<TestBackend>::default();
        OneShot::add_to(cloned, &mut commands);

        assert_eq!(commands.one_shot_commands.len(), 1);
    }

    #[test]
    fn frame_command_queue_is_fifo() {
        let mut queue = FrameCommandQueue::<TestBackend>::default();

        let mut first = FrameCommands::<TestBackend>::default();
        first.one_shot_commands.push(Box::new(TestOneShotCommand));
        queue.push(first);

        let mut second = FrameCommands::<TestBackend>::default();
        second
            .per_frame_commands
            .push(Box::new(TestPerFrameCommand));
        queue.push(second);

        let popped_first = queue.pop().expect("missing first element");
        assert_eq!(popped_first.one_shot_commands.len(), 1);
        assert_eq!(popped_first.per_frame_commands.len(), 0);

        let popped_second = queue.pop().expect("missing second element");
        assert_eq!(popped_second.one_shot_commands.len(), 0);
        assert_eq!(popped_second.per_frame_commands.len(), 1);
    }

    #[test]
    fn frame_command_queue_pop_returns_none_when_empty() {
        let mut queue = FrameCommandQueue::<TestBackend>::default();
        assert!(queue.pop().is_none());
    }
}
