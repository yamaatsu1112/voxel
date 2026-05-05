use crate::rendering::GpuMutex;
use crate::rendering::double_buffer::DoubleBuffer;
use crate::rendering::executor::Executor;
use crate::rendering::renderer_command::{FrameCommandQueue, RendererBackend, RendererCommand};
use crate::vulkan::ErasedGraphicsSubmitGroup;
use crate::vulkan::command_resource_manager::CommandResourceManager;
use crate::vulkan::compute_executor::{ComputeData, ComputeExecutor};
use crate::vulkan::context::VulkanContext;
use crate::vulkan::descriptors::DescriptorManager;
use crate::vulkan::pipeline_manager::PipelineManager;
use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::record_resource::RecordResource;
use crate::vulkan::resource_config::{
    BufferMarker, ComputePipelineMarker, DescriptorPoolMarker, DescriptorSetMarker,
    GraphicsPipelineMarker, ImageMarker, ImageViewMarker, SamplerMarker,
};
use crate::vulkan::resource_lifetime::{PerFrame as PerFrameLifetime, Persistent};
use crate::vulkan::resource_manager::ResourceManager;
use crate::vulkan::resources::{CommandResourceBuilder, RendererCommandResources};
use crate::vulkan::staging_ring_buffer::StagingRingBuffer;
use crate::vulkan::swapchain::SwapchainManager;
use crate::vulkan::sync::SyncObjects;
use crate::vulkan::vulkan_renderer_core::{
    MAX_FRAMES_IN_FLIGHT, VulkanRendererCore, acquire_mutex_state, wait_for_timeline_value,
};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};

use ash::vk;

use winit::window::Window;

pub struct VulkanRenderer {
    pub(crate) core: VulkanRendererCore,
}

impl VulkanRenderer {
    /// Create a new VulkanRenderer with Vulkan infrastructure initialized.
    /// Resource creation, descriptor setup, pipeline setup, and recordable registration
    /// are done via setup_default_renderer() after construction.
    pub fn new(window: &Arc<Window>) -> (Self, Executor) {
        let window = Arc::clone(window);
        let context = VulkanContext::new(&window);
        let swapchain = SwapchainManager::new(&context, &window);
        let mut queue_family_indices = Vec::new();
        for family in [
            context.family_indices.graphics_family.unwrap(),
            context.family_indices.compute_family.unwrap(),
            context.family_indices.transfer_family.unwrap(),
        ] {
            if !queue_family_indices.contains(&family) {
                queue_family_indices.push(family);
            }
        }
        let graphics_staging_buffers = std::array::from_fn(|_| {
            StagingRingBuffer::new(
                &context.device,
                context.physical_device,
                &context.instance,
                &queue_family_indices,
            )
        });
        let compute_staging_buffers = std::array::from_fn(|_| {
            StagingRingBuffer::new(
                &context.device,
                context.physical_device,
                &context.instance,
                &queue_family_indices,
            )
        });

        // Create descriptor manager
        let descriptor_manager =
            Arc::new(RwLock::new(DescriptorManager::new(context.device.clone())));
        let pipeline_manager = Arc::new(RwLock::new(PipelineManager::new(context.device.clone())));
        #[allow(clippy::arc_with_non_send_sync)]
        let resource_manager = Arc::new(RwLock::new(ResourceManager::new(
            context.device.clone(),
            context.instance.clone(),
            context.physical_device,
        )));
        let gpu_mutexes = Arc::new(Mutex::new(HashMap::new()));

        let mut graphics_command_resource_manager =
            CommandResourceManager::new(context.device.clone());
        let mut compute_command_resource_manager =
            CommandResourceManager::new(context.device.clone());

        let command_resources = CommandResourceBuilder::build_renderer_command_resources(
            &context,
            &mut graphics_command_resource_manager,
        );
        let compute_command_resources = CommandResourceBuilder::build_compute_command_resources(
            &context,
            &mut compute_command_resource_manager,
        );

        let sync = SyncObjects::new(&context.device, swapchain.get_image_views().len());

        let last_swapchain_extent = swapchain.get_extent();

        #[cfg(feature = "gpu-profiling")]
        let gpu_timing_manager = {
            let props = unsafe {
                context
                    .instance
                    .get_physical_device_properties(context.physical_device)
            };
            crate::vulkan::gpu_timing::GpuTimingManager::new(
                &context.device,
                props.limits.timestamp_period,
            )
        };

        let compute_data = ComputeData::new(compute_staging_buffers);
        let compute_executor = ComputeExecutor::new(
            context.device.clone(),
            compute_command_resource_manager,
            compute_command_resources,
            compute_data,
            context.queues.compute,
            context.queues.compute,
            Arc::clone(&pipeline_manager),
            Arc::clone(&descriptor_manager),
            Arc::clone(&resource_manager),
            Arc::clone(&gpu_mutexes),
            context.physical_device,
            context.instance.clone(),
            queue_family_indices,
        );

        let renderer = Self {
            core: VulkanRendererCore {
                window,
                context,
                swapchain,
                descriptor_manager,
                pipeline_manager,
                resource_manager,
                command_resource_manager: graphics_command_resource_manager,
                command_resources,
                sync,
                current_frame: 0,
                framebuffer_resized: false,
                frame_submit_groups: Vec::new(),
                graphics_record_resource: RecordResource::new(),
                on_swapchain_resize_per_frame_commands: Vec::new(),
                frame_command_buffer: None,
                per_frame_commands: std::array::from_fn(|_| Vec::new()),
                remaining_resize_executions: 0,
                last_swapchain_extent,
                graphics_staging_buffers,
                graphics_transfer_next_timeline_value: 1,
                graphics_transfer_last_submitted_values: [0; MAX_FRAMES_IN_FLIGHT],
                gpu_mutexes,
                #[cfg(feature = "gpu-profiling")]
                gpu_timing: gpu_timing_manager,
            },
        };
        (renderer, Executor::new(compute_executor))
    }

    pub(crate) fn set_frame_command_buffer(
        &mut self,
        frame_command_buffer: Arc<DoubleBuffer<FrameCommandQueue<VulkanRenderer>>>,
    ) {
        self.core.frame_command_buffer = Some(frame_command_buffer);
    }

    pub(crate) fn set_frame_submit_groups(
        &mut self,
        submit_groups: Vec<Box<dyn ErasedGraphicsSubmitGroup>>,
    ) {
        self.core.set_frame_submit_groups(submit_groups);
    }

    pub(crate) fn set_graphics_record_resource(&mut self, resource: RecordResource) {
        self.core.set_graphics_record_resource(resource);
    }

    pub fn register_on_swapchain_resize_per_frame(
        &mut self,
        command: Box<dyn RendererCommand<VulkanRenderer>>,
    ) {
        self.core
            .on_swapchain_resize_per_frame_commands
            .push(command);
    }

    pub fn execute_per_frame_resize_commands(&mut self) {
        let commands = std::mem::take(&mut self.core.on_swapchain_resize_per_frame_commands);
        for command in &commands {
            command.execute(self);
        }
        self.core.on_swapchain_resize_per_frame_commands = commands;
    }

    pub fn flush_pending_commands(&mut self) {
        let Some(frame_command_buffer) = self.core.frame_command_buffer.as_ref().map(Arc::clone)
        else {
            return;
        };

        let mut queue = frame_command_buffer.swap_and_read();
        self.execute_frame_command_queue(&mut queue);
    }

    pub fn execute_frame_command_queue(&mut self, queue: &mut FrameCommandQueue<VulkanRenderer>) {
        while let Some(mut frame_commands) = queue.pop() {
            for command in frame_commands.one_shot_commands.drain(..) {
                command.execute(self);
            }

            for command in frame_commands.per_frame_commands.drain(..) {
                for i in 0..MAX_FRAMES_IN_FLIGHT {
                    self.core.per_frame_commands[i].push(command.clone_box());
                }
            }
        }
    }

    pub fn enqueue_frame_command_queue(&mut self, queue: &mut FrameCommandQueue<VulkanRenderer>) {
        let Some(frame_command_buffer) = self.core.frame_command_buffer.as_ref().map(Arc::clone)
        else {
            return;
        };
        let mut write_buffer = frame_command_buffer.write_buffer();
        while let Some(frame_commands) = queue.pop() {
            write_buffer.push(frame_commands);
        }
    }

    pub(crate) fn process_per_frame_commands(&mut self) {
        let commands = std::mem::take(&mut self.core.per_frame_commands[self.core.current_frame]);
        for command in &commands {
            command.execute(self);
        }
    }

    pub fn process_frame(&mut self) {
        let current_frame = self.core.current_frame;
        let in_flight_fence = self.core.sync.in_flight_fences[current_frame];
        unsafe {
            let _ = self
                .core
                .context
                .device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX);
            let _ = self.core.context.device.reset_fences(&[in_flight_fence]);
        }
        wait_for_timeline_value(
            &self.core.context.device,
            self.core.sync.graphics_transfer_timeline,
            self.core.graphics_transfer_last_submitted_values[current_frame],
        );

        #[cfg(feature = "gpu-profiling")]
        {
            self.core
                .gpu_timing
                .collect_results(&self.core.context.device, current_frame);
            self.core.gpu_timing.maybe_log();
        }

        self.core
            .command_resource_manager
            .reset_command_pool(self.core.command_resources.graphics_pool_ids[current_frame]);
        self.core.command_resource_manager.reset_command_pool(
            self.core.command_resources.graphics_transfer_pool_ids[current_frame],
        );

        self.core.graphics_staging_buffers[current_frame].reset(&self.core.context.device);

        self.flush_pending_commands();
        self.process_per_frame_commands();
        self.draw_frame();
        self.core.current_frame = (self.core.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn draw_frame(&mut self) {
        if self.core.swapchain.get_extent() != self.core.last_swapchain_extent {
            self.core.remaining_resize_executions = MAX_FRAMES_IN_FLIGHT;
            self.core.last_swapchain_extent = self.core.swapchain.get_extent();
        }

        if self.core.remaining_resize_executions > 0 {
            self.execute_per_frame_resize_commands();
            self.core.remaining_resize_executions -= 1;
        }

        let core = &mut self.core;
        let current_frame = core.current_frame;
        let image_available_semaphore = core.sync.image_available_semaphores[current_frame];
        let in_flight_fence = core.sync.in_flight_fences[current_frame];
        let swapchain_loader = core.swapchain.get_swapchain_loader();
        let swapchain = core.swapchain.get_swapchain();

        unsafe {
            let result = swapchain_loader.acquire_next_image(
                swapchain,
                u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            );
            let (image_index, _suboptimal) = match result {
                Ok((idx, suboptimal)) => (idx, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    let submit_info = vk::SubmitInfo {
                        s_type: vk::StructureType::SUBMIT_INFO,
                        p_next: ptr::null(),
                        wait_semaphore_count: 0,
                        p_wait_semaphores: ptr::null(),
                        p_wait_dst_stage_mask: ptr::null(),
                        command_buffer_count: 0,
                        p_command_buffers: ptr::null(),
                        signal_semaphore_count: 0,
                        p_signal_semaphores: ptr::null(),
                        _marker: PhantomData,
                    };
                    let _ = core.context.device.queue_submit(
                        core.context.queues.graphics,
                        &[submit_info],
                        in_flight_fence,
                    );
                    core.swapchain.recreate(&core.context, &core.window);
                    return;
                }
                Err(e) => {
                    panic!("Failed to acquire swapchain image: {:?}", e);
                }
            };
            let submit_groups = &core.frame_submit_groups;
            if submit_groups.is_empty() {
                let signal_semaphore = core.sync.render_finished_semaphores[image_index as usize];
                let wait_stage_masks = [vk::PipelineStageFlags::TOP_OF_PIPE];
                let submit_info = vk::SubmitInfo {
                    s_type: vk::StructureType::SUBMIT_INFO,
                    p_next: ptr::null(),
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &image_available_semaphore,
                    p_wait_dst_stage_mask: wait_stage_masks.as_ptr(),
                    command_buffer_count: 0,
                    p_command_buffers: ptr::null(),
                    signal_semaphore_count: 1,
                    p_signal_semaphores: &signal_semaphore,
                    _marker: PhantomData,
                };
                let _ = core.context.device.queue_submit(
                    core.context.queues.graphics,
                    &[submit_info],
                    in_flight_fence,
                );
            }
            let last_group_index = submit_groups.len().saturating_sub(1);

            #[cfg(feature = "gpu-profiling")]
            if !submit_groups.is_empty() && core.gpu_timing.query_count(current_frame) == 0 {
                let reset_cb = core.command_resource_manager.get_free_command_buffer(
                    core.command_resources.graphics_pool_ids[current_frame],
                    vk::CommandBufferLevel::PRIMARY,
                );
                let begin_info = vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: ptr::null(),
                    _marker: PhantomData,
                };
                core.context
                    .device
                    .begin_command_buffer(reset_cb, &begin_info)
                    .expect("Failed to begin reset command buffer");
                core.context.device.cmd_reset_query_pool(
                    reset_cb,
                    core.gpu_timing.query_pool(current_frame),
                    0,
                    crate::vulkan::gpu_timing::GpuTimingManager::max_timestamps_per_frame(),
                );
                core.context
                    .device
                    .end_command_buffer(reset_cb)
                    .expect("Failed to end reset command buffer");
                let submit_info = vk::SubmitInfo {
                    s_type: vk::StructureType::SUBMIT_INFO,
                    p_next: ptr::null(),
                    wait_semaphore_count: 0,
                    p_wait_semaphores: ptr::null(),
                    p_wait_dst_stage_mask: ptr::null(),
                    command_buffer_count: 1,
                    p_command_buffers: &reset_cb,
                    signal_semaphore_count: 0,
                    p_signal_semaphores: ptr::null(),
                    _marker: PhantomData,
                };
                let _ = core.context.device.queue_submit(
                    core.context.queues.graphics,
                    &[submit_info],
                    vk::Fence::null(),
                );
            }

            for (group_index, submit_group) in submit_groups.iter().enumerate() {
                let pipeline_manager = core.pipeline_manager.read().unwrap();
                let descriptor_manager = core.descriptor_manager.read().unwrap();
                let resource_manager = core.resource_manager.read().unwrap();
                let command_buffer = Self::record_graphics_command_buffer(
                    &core.context.device,
                    &pipeline_manager,
                    &descriptor_manager,
                    &resource_manager,
                    &mut core.command_resource_manager,
                    &core.command_resources,
                    &core.swapchain,
                    &core.graphics_record_resource,
                    current_frame,
                    image_index as usize,
                    submit_group.as_ref(),
                    #[cfg(feature = "gpu-profiling")]
                    &mut core.gpu_timing,
                );

                let mut wait_semaphores = Vec::new();
                let mut wait_stages = Vec::new();
                let mut wait_values = Vec::new();
                let mut signal_semaphores = Vec::new();
                let mut signal_values = Vec::new();

                for mutex_id in submit_group.mutex_ids() {
                    let mut gpu_mutexes = core.gpu_mutexes.lock().unwrap();
                    let (semaphore, wait_value, signal_value) =
                        acquire_mutex_state(&mut gpu_mutexes, &core.context.device, mutex_id);
                    wait_semaphores.push(semaphore);
                    wait_stages.push(vk::PipelineStageFlags::TRANSFER);
                    wait_values.push(wait_value);
                    signal_semaphores.push(semaphore);
                    signal_values.push(signal_value);
                }

                let signal_semaphore = core.sync.render_finished_semaphores[image_index as usize];
                let has_signal = group_index == last_group_index;
                if has_signal {
                    wait_semaphores.push(image_available_semaphore);
                    wait_stages.push(vk::PipelineStageFlags::TOP_OF_PIPE);
                    wait_values.push(0);
                    signal_semaphores.push(signal_semaphore);
                    signal_values.push(0);
                }
                let has_timeline = !signal_values.is_empty()
                    && signal_values.iter().any(|&v| v != 0)
                    || !wait_values.is_empty() && wait_values.iter().any(|&v| v != 0);
                let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
                    s_type: vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO,
                    p_next: ptr::null(),
                    wait_semaphore_value_count: wait_values.len() as u32,
                    p_wait_semaphore_values: wait_values.as_ptr(),
                    signal_semaphore_value_count: signal_values.len() as u32,
                    p_signal_semaphore_values: signal_values.as_ptr(),
                    _marker: PhantomData,
                };
                let submit_info = vk::SubmitInfo {
                    s_type: vk::StructureType::SUBMIT_INFO,
                    p_next: if has_timeline {
                        (&timeline_submit_info as *const vk::TimelineSemaphoreSubmitInfo).cast()
                    } else {
                        ptr::null()
                    },
                    wait_semaphore_count: wait_semaphores.len() as u32,
                    p_wait_semaphores: wait_semaphores.as_ptr(),
                    p_wait_dst_stage_mask: wait_stages.as_ptr(),
                    command_buffer_count: 1,
                    p_command_buffers: &command_buffer,
                    signal_semaphore_count: signal_semaphores.len() as u32,
                    p_signal_semaphores: signal_semaphores.as_ptr(),
                    _marker: PhantomData,
                };
                let submit_fence = if has_signal {
                    in_flight_fence
                } else {
                    vk::Fence::null()
                };
                let _ = core.context.device.queue_submit(
                    core.context.queues.graphics,
                    &[submit_info],
                    submit_fence,
                );
            }

            let present_info = vk::PresentInfoKHR {
                s_type: vk::StructureType::PRESENT_INFO_KHR,
                p_next: ptr::null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: &core.sync.render_finished_semaphores[image_index as usize],
                p_image_indices: &image_index,
                p_results: ptr::null_mut(),
                p_swapchains: &swapchain,
                swapchain_count: 1,
                _marker: PhantomData,
            };
            let swapchain_loader = core.swapchain.get_swapchain_loader();
            let result = swapchain_loader.queue_present(core.context.queues.present, &present_info);
            match result {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    core.framebuffer_resized = false;
                    core.swapchain.recreate(&core.context, &core.window);
                }
                Err(e) => {
                    panic!("Failed to present swapchain image: {:?}", e);
                }
            }

            if core.framebuffer_resized {
                core.framebuffer_resized = false;
                core.swapchain.recreate(&core.context, &core.window);
            }
        }
    }

    fn cleanup(&mut self) {
        // Cleanup core-owned gpu_mutexes
        self.core.cleanup();
    }

    pub fn get_window_size(&self) -> (u32, u32) {
        let extent = self.core.swapchain.get_extent();
        (extent.width, extent.height)
    }

    pub fn read_buffer_data<T: BufferMarker<Lifetime = Persistent>, D: Copy, const N: usize>(
        &self,
        index: usize,
    ) -> Option<[D; N]> {
        self.core.read_buffer_data::<T, D, N>(index)
    }

    fn record_graphics_command_buffer(
        device: &ash::Device,
        pipeline_manager: &PipelineManager,
        descriptor_manager: &DescriptorManager,
        resource_manager: &ResourceManager,
        command_resource_manager: &mut CommandResourceManager,
        command_resources: &RendererCommandResources,
        swapchain: &SwapchainManager,
        record_resource: &RecordResource,
        current_frame: usize,
        image_index: usize,
        submit_group: &dyn ErasedGraphicsSubmitGroup,
        #[cfg(feature = "gpu-profiling")]
        gpu_timing: &mut crate::vulkan::gpu_timing::GpuTimingManager,
    ) -> vk::CommandBuffer {
        let command_buffer = command_resource_manager.get_free_command_buffer(
            command_resources.graphics_pool_ids[current_frame],
            vk::CommandBufferLevel::PRIMARY,
        );

        let mut context = VkGraphicsRecordContext::new(
            device,
            command_buffer,
            current_frame,
            image_index,
            pipeline_manager,
            descriptor_manager,
            resource_manager,
            command_resource_manager,
            command_resources,
            swapchain,
            record_resource,
            #[cfg(feature = "gpu-profiling")]
            gpu_timing,
        );

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: ptr::null(),
            _marker: PhantomData,
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin command buffer");
        }

        submit_group.record(&mut context);

        unsafe {
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer");
        }

        command_buffer
    }
}

impl RendererBackend for VulkanRenderer {
    fn create_buffers<T: BufferMarker + Send + Sync + 'static>(&mut self) {
        self.core.create_buffers::<T>();
    }

    fn create_images<T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static>(
        &mut self,
        size: Option<(u32, u32)>,
    ) {
        self.core.create_images::<T>(size);
    }

    fn create_image_views<T: ImageViewMarker<Lifetime = Persistent> + Send + Sync + 'static>(
        &mut self,
    ) {
        self.core.create_image_views::<T>();
    }

    fn create_sampler<T: SamplerMarker + Send + Sync + 'static>(&mut self) {
        self.core.create_sampler::<T>();
    }

    fn create_descriptor_pool<P: DescriptorPoolMarker + Send + Sync + 'static>(
        &mut self,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
        flags: vk::DescriptorPoolCreateFlags,
    ) {
        self.core
            .create_descriptor_pool::<P>(pool_sizes, max_sets, flags);
    }

    fn create_descriptor_sets<T: DescriptorSetMarker + Send + Sync + 'static>(&mut self) {
        self.core.create_descriptor_sets::<T>();
    }

    fn create_graphics_pipeline<T: GraphicsPipelineMarker + Send + Sync + 'static>(&mut self) {
        self.core.create_graphics_pipeline::<T>();
    }

    fn create_compute_pipeline<T: ComputePipelineMarker + Send + Sync + 'static>(&mut self) {
        self.core.create_compute_pipeline::<T>();
    }

    fn bind_buffer<
        D: DescriptorSetMarker<Lifetime = Persistent> + Send + Sync + 'static,
        B: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
    >(
        &mut self,
        binding: u32,
        array_index: u32,
        buffer_index: usize,
        count: usize,
        descriptor_type: vk::DescriptorType,
    ) {
        self.core
            .bind_buffer::<D, B>(binding, array_index, buffer_index, count, descriptor_type);
    }

    fn bind_per_frame_buffer<
        D: DescriptorSetMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
        B: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        binding: u32,
        array_index: u32,
        buffer_index: usize,
        count: usize,
        descriptor_type: vk::DescriptorType,
    ) {
        self.core.bind_per_frame_buffer::<D, B>(
            binding,
            array_index,
            buffer_index,
            count,
            descriptor_type,
        );
    }

    fn bind_image<
        D: DescriptorSetMarker<Lifetime = Persistent> + Send + Sync + 'static,
        V: ImageViewMarker<Lifetime = Persistent> + Send + Sync + 'static,
        S: SamplerMarker + Send + Sync + 'static,
    >(
        &mut self,
        binding: u32,
        array_index: u32,
        image_index: usize,
        count: usize,
    ) {
        self.core
            .bind_image::<D, V, S>(binding, array_index, image_index, count);
    }

    fn upload_to_buffer<
        M: GpuMutex + 'static,
        T: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
        D: Clone + Send + Sync + 'static,
    >(
        &mut self,
        index: usize,
        data: &[D],
        dst_byte_offset: u64,
    ) {
        self.core
            .upload_to_buffer::<M, T, D>(index, data, dst_byte_offset);
    }

    fn upload_to_per_frame_buffer<
        M: GpuMutex + 'static,
        T: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
        D: Clone + Send + Sync + 'static,
    >(
        &mut self,
        index: usize,
        data: &[D],
    ) {
        self.core.upload_to_per_frame_buffer::<M, T, D>(index, data);
    }

    fn create_per_frame_images<
        T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        size: Option<(u32, u32)>,
        frame_index: usize,
    ) {
        self.core.create_per_frame_images::<T>(size, frame_index);
    }

    fn create_per_frame_image_views<
        T: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        frame_index: usize,
    ) {
        self.core.create_per_frame_image_views::<T>(frame_index);
    }

    fn bind_per_frame_image<
        D: DescriptorSetMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
        V: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
        S: SamplerMarker + Send + Sync + 'static,
    >(
        &mut self,
        binding: u32,
        array_index: u32,
        image_index: usize,
        count: usize,
    ) {
        self.core
            .bind_per_frame_image::<D, V, S>(binding, array_index, image_index, count);
    }

    fn destroy_per_frame_images<
        T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        frame_index: usize,
    ) {
        self.core.destroy_per_frame_images::<T>(frame_index);
    }

    fn destroy_per_frame_image_views<
        T: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        frame_index: usize,
    ) {
        self.core.destroy_per_frame_image_views::<T>(frame_index);
    }

    fn transition_image_layout<
        M: GpuMutex + 'static,
        T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    >(
        &mut self,
        index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        self.core
            .transition_image_layout::<M, T>(index, old_layout, new_layout, aspect_mask);
    }

    fn transition_per_frame_image_layout<
        M: GpuMutex + 'static,
        T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    >(
        &mut self,
        frame_index: usize,
        image_index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        self.core.transition_per_frame_image_layout::<M, T>(
            frame_index,
            image_index,
            old_layout,
            new_layout,
            aspect_mask,
        );
    }

    fn current_frame_index(&self) -> usize {
        self.core.current_frame_index()
    }

    fn upload_to_image<
        M: GpuMutex + 'static,
        T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    >(
        &mut self,
        image_index: usize,
        data: &[u8],
        width: u32,
        height: u32,
        depth: u32,
    ) {
        self.core
            .upload_to_image::<M, T>(image_index, data, width, height, depth);
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.core.context.device.device_wait_idle();
        }
        self.cleanup();
    }
}
