use crate::rendering::GpuMutex;
use crate::rendering::double_buffer::DoubleBuffer;
use crate::rendering::renderer_command::{FrameCommandQueue, RendererCommand};
use crate::vulkan::ErasedGraphicsSubmitGroup;
use crate::vulkan::command_resource_manager::CommandResourceManager;
use crate::vulkan::context::VulkanContext;
use crate::vulkan::debug::VALIDATION;
use crate::vulkan::descriptors::DescriptorManager;
use crate::vulkan::gpu_mutex::GpuMutexState;
use crate::vulkan::pipeline_manager::PipelineManager;
use crate::vulkan::record_resource::RecordResource;
use crate::vulkan::resource_config::{
    BufferId, BufferMarker, ComputePipelineId, ComputePipelineMarker, DescriptorPoolMarker,
    DescriptorSetId, DescriptorSetMarker, GraphicsPipelineId, GraphicsPipelineMarker, ImageConfig,
    ImageId, ImageMarker, ImageSizeFormat, ImageViewId, ImageViewMarker, SamplerId, SamplerMarker,
    buffer_id, compute_pipeline_id, descriptor_pool_id, graphics_pipeline_id, image_id,
    image_view_id, per_frame_buffer_id, per_frame_image_id, per_frame_image_view_id, sampler_id,
};
use crate::vulkan::resource_lifetime::{PerFrame, Persistent};
use crate::vulkan::resource_manager::ResourceManager;
use crate::vulkan::resources::RendererCommandResources;
use crate::vulkan::staging_ring_buffer::StagingRingBuffer;
use crate::vulkan::swapchain::SwapchainManager;
use crate::vulkan::sync::SyncObjects;
use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};

use ash::vk;

use winit::window::Window;

// Constants
const _WINDOW_WIDTH: u32 = 800;
const _WINDOW_HEIGHT: u32 = 600;
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub const MAX_COMPUTE_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRendererCore {
    pub(crate) window: Arc<Window>,
    pub(crate) context: VulkanContext,
    pub(crate) swapchain: SwapchainManager,
    pub(crate) descriptor_manager: Arc<RwLock<DescriptorManager>>,
    pub(crate) pipeline_manager: Arc<RwLock<PipelineManager>>,
    pub(crate) resource_manager: Arc<RwLock<ResourceManager>>,
    pub(crate) command_resource_manager: CommandResourceManager,
    pub(crate) command_resources: RendererCommandResources,
    pub(crate) sync: SyncObjects,
    pub(crate) current_frame: usize,
    pub(crate) framebuffer_resized: bool,
    pub(crate) frame_submit_groups: Vec<Box<dyn ErasedGraphicsSubmitGroup>>,
    pub(crate) graphics_record_resource: RecordResource,
    pub(super) on_swapchain_resize_per_frame_commands:
        Vec<Box<dyn RendererCommand<super::vulkan_renderer::VulkanRenderer>>>,
    pub(super) frame_command_buffer:
        Option<Arc<DoubleBuffer<FrameCommandQueue<super::vulkan_renderer::VulkanRenderer>>>>,
    pub(super) per_frame_commands: [Vec<
        Box<dyn RendererCommand<super::vulkan_renderer::VulkanRenderer>>,
    >; MAX_FRAMES_IN_FLIGHT],
    pub(crate) remaining_resize_executions: usize,
    pub(crate) last_swapchain_extent: vk::Extent2D,
    pub(crate) graphics_staging_buffers: [StagingRingBuffer; MAX_FRAMES_IN_FLIGHT],
    pub(crate) graphics_transfer_next_timeline_value: u64,
    pub(crate) graphics_transfer_last_submitted_values: [u64; MAX_FRAMES_IN_FLIGHT],
    pub(crate) gpu_mutexes: Arc<Mutex<HashMap<TypeId, GpuMutexState>>>,
    #[cfg(feature = "gpu-profiling")]
    pub(crate) gpu_timing: crate::vulkan::gpu_timing::GpuTimingManager,
}

impl VulkanRendererCore {
    pub(crate) fn get_queue_family_indices(&self) -> Vec<u32> {
        let mut family_indices = Vec::new();
        for family in [
            self.context.family_indices.graphics_family.unwrap(),
            self.context.family_indices.compute_family.unwrap(),
            self.context.family_indices.transfer_family.unwrap(),
        ] {
            if !family_indices.contains(&family) {
                family_indices.push(family);
            }
        }
        family_indices
    }

    pub(crate) fn acquire_mutex(&mut self, type_id: TypeId) -> (vk::Semaphore, u64, u64) {
        let mut gpu_mutexes = self.gpu_mutexes.lock().unwrap();
        acquire_mutex_state(&mut gpu_mutexes, &self.context.device, type_id)
    }

    pub fn set_framebuffer_resized(&mut self, resized: bool) {
        self.framebuffer_resized = resized;
    }

    pub(crate) fn current_frame_index(&self) -> usize {
        self.current_frame
    }

    pub fn set_graphics_record_resource(&mut self, value: RecordResource) {
        self.graphics_record_resource = value;
    }

    // ========================================================================
    // Pipeline APIs
    // ========================================================================

    /// Create a graphics pipeline from marker trait config, using descriptor_set_layouts from it.
    pub fn create_graphics_pipeline<T: GraphicsPipelineMarker>(&mut self) {
        let config = &T::CONFIG;
        let layouts: Vec<ash::vk::DescriptorSetLayout> = {
            let mut descriptor_manager = self.descriptor_manager.write().unwrap();
            config
                .descriptor_set_layouts
                .iter()
                .map(|bindings| {
                    let id = descriptor_manager.create_descriptor_set_layout(bindings);
                    descriptor_manager
                        .get_descriptor_set_layout(id)
                        .expect("Failed to get descriptor set layout")
                })
                .collect()
        };

        let old_id = graphics_pipeline_id::<T>();
        if old_id != GraphicsPipelineId::INVALID {
            self.pipeline_manager
                .write()
                .unwrap()
                .destroy_graphics_pipeline(old_id);
        }

        let swapchain_format = self.swapchain.get_format();
        let new_id = self
            .pipeline_manager
            .write()
            .unwrap()
            .create_graphics_pipeline(config, &layouts, swapchain_format);
        new_id.store(T::id());
    }

    /// Create a compute pipeline from marker trait config, using descriptor_set_layouts from it.
    pub fn create_compute_pipeline<T: ComputePipelineMarker>(&mut self) {
        let config = &T::CONFIG;
        let layouts: Vec<ash::vk::DescriptorSetLayout> = {
            let mut descriptor_manager = self.descriptor_manager.write().unwrap();
            config
                .descriptor_set_layouts
                .iter()
                .map(|bindings| {
                    let id = descriptor_manager.create_descriptor_set_layout(bindings);
                    descriptor_manager
                        .get_descriptor_set_layout(id)
                        .expect("Failed to get descriptor set layout")
                })
                .collect()
        };

        let old_id = compute_pipeline_id::<T>();
        if old_id != ComputePipelineId::INVALID {
            self.pipeline_manager
                .write()
                .unwrap()
                .destroy_compute_pipeline(old_id);
        }

        let new_id = self
            .pipeline_manager
            .write()
            .unwrap()
            .create_compute_pipeline(config, &layouts);
        new_id.store(T::id());
    }

    // ========================================================================
    // Descriptor Management APIs
    // ========================================================================

    /// Create a descriptor pool with the given pool sizes and configuration.
    pub fn create_descriptor_pool<P: DescriptorPoolMarker>(
        &mut self,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
        flags: vk::DescriptorPoolCreateFlags,
    ) {
        let pool_id = self
            .descriptor_manager
            .write()
            .unwrap()
            .create_descriptor_pool(pool_sizes, max_sets, flags);
        pool_id.store(P::id());
    }

    /// Create descriptor sets identified by marker type T.
    pub fn create_descriptor_sets<T: DescriptorSetMarker>(&mut self) {
        let mut descriptor_manager = self.descriptor_manager.write().unwrap();
        let layout_id = descriptor_manager.create_descriptor_set_layout(T::CONFIG.layout_bindings);
        let pool_id = descriptor_pool_id::<T::Pool>();
        let ids = T::ids_slice();

        for atomic_id in ids.iter() {
            let set_id = descriptor_manager.create_descriptor_set(pool_id, layout_id);
            set_id.store(atomic_id);
        }
    }

    // ========================================================================
    // Resource Creation APIs
    // ========================================================================

    /// Create buffers identified by marker type T.
    /// Works for both Persistent and PerFrame buffers.
    pub fn create_buffers<T: BufferMarker>(&mut self) {
        let queue_family_indices = self.get_queue_family_indices();
        let ids = T::ids_slice();

        for atomic_id in ids.iter() {
            let id = BufferId::load(atomic_id);
            if id != BufferId::INVALID {
                self.resource_manager.write().unwrap().destroy_buffer(id);
                BufferId::INVALID.store(atomic_id);
            }
        }

        for atomic_id in ids.iter() {
            let new_id = self
                .resource_manager
                .write()
                .unwrap()
                .create_buffer(&T::CONFIG, &queue_family_indices);
            new_id.store(atomic_id);
        }
    }

    pub fn get_buffer<T: BufferMarker<Lifetime = Persistent>>(
        &self,
        index: usize,
    ) -> Option<vk::Buffer> {
        let id = buffer_id::<T>(index)?;
        self.resource_manager.read().unwrap().get_buffer(id)
    }

    pub fn get_per_frame_buffer<T: BufferMarker<Lifetime = PerFrame>>(
        &self,
        frame: usize,
        index: usize,
    ) -> Option<vk::Buffer> {
        let id = per_frame_buffer_id::<T>(frame, index)?;
        self.resource_manager.read().unwrap().get_buffer(id)
    }

    pub fn get_buffer_size<T: BufferMarker<Lifetime = Persistent>>(
        &self,
        index: usize,
    ) -> Option<u64> {
        let id = buffer_id::<T>(index)?;
        self.resource_manager.read().unwrap().get_buffer_size(id)
    }

    pub fn get_per_frame_buffer_size<T: BufferMarker<Lifetime = PerFrame>>(
        &self,
        frame: usize,
        index: usize,
    ) -> Option<u64> {
        let id = per_frame_buffer_id::<T>(frame, index)?;
        self.resource_manager.read().unwrap().get_buffer_size(id)
    }

    pub fn is_buffer_host_visible<T: BufferMarker<Lifetime = Persistent>>(
        &self,
        index: usize,
    ) -> bool {
        if let Some(id) = buffer_id::<T>(index) {
            self.resource_manager
                .read()
                .unwrap()
                .is_buffer_host_visible(id)
        } else {
            false
        }
    }

    pub fn is_per_frame_buffer_host_visible<T: BufferMarker<Lifetime = PerFrame>>(
        &self,
        frame: usize,
        index: usize,
    ) -> bool {
        if let Some(id) = per_frame_buffer_id::<T>(frame, index) {
            self.resource_manager
                .read()
                .unwrap()
                .is_buffer_host_visible(id)
        } else {
            false
        }
    }

    pub fn copy_data_to_buffer<T: BufferMarker<Lifetime = Persistent>, D>(
        &self,
        index: usize,
        dst_byte_offset: u64,
        data: &[D],
    ) {
        let id = buffer_id::<T>(index).expect("Invalid buffer index for copy_data_to_buffer");
        self.resource_manager
            .read()
            .unwrap()
            .copy_data_to_buffer(id, dst_byte_offset, data);
    }

    pub fn copy_data_to_per_frame_buffer<T: BufferMarker<Lifetime = PerFrame>, D>(
        &self,
        frame: usize,
        index: usize,
        data: &[D],
    ) {
        let id = per_frame_buffer_id::<T>(frame, index)
            .expect("Invalid per-frame buffer index for copy_data_to_per_frame_buffer");
        self.resource_manager
            .read()
            .unwrap()
            .copy_data_to_buffer(id, 0, data);
    }

    pub fn read_buffer_data<T: BufferMarker<Lifetime = Persistent>, D: Copy, const N: usize>(
        &self,
        index: usize,
    ) -> Option<[D; N]> {
        let id = buffer_id::<T>(index)?;
        self.resource_manager.read().unwrap().read_buffer_data(id)
    }

    /// Create persistent images identified by marker type T.
    pub fn create_images<T: ImageMarker<Lifetime = Persistent>>(
        &mut self,
        size: Option<(u32, u32)>,
    ) {
        let queue_family_indices = self.get_queue_family_indices();
        let swapchain_extent = self.swapchain.get_extent();
        let mut config = T::CONFIG;
        if let Some((width, height)) = size {
            assert!(
                matches!(config.size, ImageSizeFormat::Dynamic),
                "size override is only allowed for ImageSizeFormat::Dynamic"
            );
            config = ImageConfig {
                size: ImageSizeFormat::Fixed { width, height },
                ..config
            };
        }

        let ids = T::ids_slice();
        for atomic_id in ids.iter() {
            let id = ImageId::load(atomic_id);
            if id != ImageId::INVALID {
                self.resource_manager.write().unwrap().destroy_image(id);
                ImageId::INVALID.store(atomic_id);
            }
        }

        for atomic_id in ids.iter() {
            let new_id = self.resource_manager.write().unwrap().create_image(
                &config,
                swapchain_extent,
                &queue_family_indices,
            );
            new_id.store(atomic_id);
        }
    }

    /// Create per-frame images identified by marker type T for a specific frame.
    pub fn create_per_frame_images<T: ImageMarker<Lifetime = PerFrame>>(
        &mut self,
        size: Option<(u32, u32)>,
        frame_index: usize,
    ) {
        let queue_family_indices = self.get_queue_family_indices();
        let swapchain_extent = self.swapchain.get_extent();
        let mut config = T::CONFIG;
        if let Some((width, height)) = size {
            assert!(
                matches!(config.size, ImageSizeFormat::Dynamic),
                "size override is only allowed for ImageSizeFormat::Dynamic"
            );
            config = ImageConfig {
                size: ImageSizeFormat::Fixed { width, height },
                ..config
            };
        }

        let ids = T::ids_slice();
        let start = frame_index * T::COUNT;
        let end = start + T::COUNT;

        for atomic_id in ids[start..end].iter() {
            let id = ImageId::load(atomic_id);
            if id != ImageId::INVALID {
                self.resource_manager.write().unwrap().destroy_image(id);
                ImageId::INVALID.store(atomic_id);
            }
        }

        for atomic_id in ids[start..end].iter() {
            let new_id = self.resource_manager.write().unwrap().create_image(
                &config,
                swapchain_extent,
                &queue_family_indices,
            );
            new_id.store(atomic_id);
        }
    }

    /// Create persistent image views identified by marker type T.
    pub fn create_image_views<T: ImageViewMarker<Lifetime = Persistent>>(&mut self) {
        let ids = T::ids_slice();
        let image_ids = T::Image::ids_slice();

        for atomic_id in ids.iter() {
            let id = ImageViewId::load(atomic_id);
            if id != ImageViewId::INVALID {
                self.resource_manager
                    .write()
                    .unwrap()
                    .destroy_image_view(id);
                ImageViewId::INVALID.store(atomic_id);
            }
        }

        for (i, atomic_id) in ids.iter().enumerate() {
            let image_id = ImageId::load(&image_ids[i]);
            let mut resource_manager = self.resource_manager.write().unwrap();
            let image = resource_manager
                .get_image(image_id)
                .expect("Referenced image not found for image view creation");
            let new_id = resource_manager.create_image_view(&T::CONFIG, image);
            new_id.store(atomic_id);
        }
    }

    /// Create per-frame image views identified by marker type T for a specific frame.
    pub fn create_per_frame_image_views<T: ImageViewMarker<Lifetime = PerFrame>>(
        &mut self,
        frame_index: usize,
    ) {
        let ids = T::ids_slice();
        let image_ids = T::Image::ids_slice();
        let start = frame_index * T::COUNT;
        let end = start + T::COUNT;

        for atomic_id in ids[start..end].iter() {
            let id = ImageViewId::load(atomic_id);
            if id != ImageViewId::INVALID {
                self.resource_manager
                    .write()
                    .unwrap()
                    .destroy_image_view(id);
                ImageViewId::INVALID.store(atomic_id);
            }
        }

        for (i, atomic_id) in ids[start..end].iter().enumerate() {
            let image_id = ImageId::load(&image_ids[start + i]);
            let mut resource_manager = self.resource_manager.write().unwrap();
            let image = resource_manager
                .get_image(image_id)
                .expect("Referenced per-frame image not found for image view creation");
            let new_id = resource_manager.create_image_view(&T::CONFIG, image);
            new_id.store(atomic_id);
        }
    }

    pub fn destroy_per_frame_images<T: ImageMarker<Lifetime = PerFrame>>(
        &mut self,
        frame_index: usize,
    ) {
        let ids = T::ids_slice();
        let start = frame_index * T::COUNT;
        let end = start + T::COUNT;

        for atomic_id in ids[start..end].iter() {
            let id = ImageId::load(atomic_id);
            if id != ImageId::INVALID {
                self.resource_manager.write().unwrap().destroy_image(id);
                ImageId::INVALID.store(atomic_id);
            }
        }
    }

    pub fn destroy_per_frame_image_views<T: ImageViewMarker<Lifetime = PerFrame>>(
        &mut self,
        frame_index: usize,
    ) {
        let ids = T::ids_slice();
        let start = frame_index * T::COUNT;
        let end = start + T::COUNT;

        for atomic_id in ids[start..end].iter() {
            let id = ImageViewId::load(atomic_id);
            if id != ImageViewId::INVALID {
                self.resource_manager
                    .write()
                    .unwrap()
                    .destroy_image_view(id);
                ImageViewId::INVALID.store(atomic_id);
            }
        }
    }

    pub fn get_image<T: ImageMarker<Lifetime = Persistent>>(
        &self,
        index: usize,
    ) -> Option<vk::Image> {
        let id = image_id::<T>(index)?;
        self.resource_manager.read().unwrap().get_image(id)
    }

    pub fn get_per_frame_image<T: ImageMarker<Lifetime = PerFrame>>(
        &self,
        frame: usize,
        index: usize,
    ) -> Option<vk::Image> {
        let id = per_frame_image_id::<T>(frame, index)?;
        self.resource_manager.read().unwrap().get_image(id)
    }

    pub fn get_image_extent<T: ImageMarker<Lifetime = Persistent>>(
        &self,
        index: usize,
    ) -> Option<vk::Extent3D> {
        let id = image_id::<T>(index)?;
        self.resource_manager.read().unwrap().get_image_extent(id)
    }

    pub fn get_image_view<T: ImageViewMarker<Lifetime = Persistent>>(
        &self,
        index: usize,
    ) -> Option<vk::ImageView> {
        let id = image_view_id::<T>(index)?;
        self.resource_manager.read().unwrap().get_image_view(id)
    }

    pub fn get_per_frame_image_view<T: ImageViewMarker<Lifetime = PerFrame>>(
        &self,
        frame: usize,
        index: usize,
    ) -> Option<vk::ImageView> {
        let id = per_frame_image_view_id::<T>(frame, index)?;
        self.resource_manager.read().unwrap().get_image_view(id)
    }

    pub fn get_sampler<T: SamplerMarker>(&self) -> Option<vk::Sampler> {
        let id = sampler_id::<T>();
        self.resource_manager.read().unwrap().get_sampler(id)
    }

    // ========================================================================
    // Upload APIs
    // ========================================================================

    /// Upload data to a buffer.
    /// - HOST_VISIBLE: Direct memory copy
    /// - DEVICE_LOCAL: Creates internal staging buffer, transfers data via GPU, then destroys staging
    pub fn upload_to_buffer<M: GpuMutex + 'static, T: BufferMarker<Lifetime = Persistent>, D>(
        &mut self,
        index: usize,
        data: &[D],
        dst_byte_offset: u64,
    ) {
        let is_host_visible = self.is_buffer_host_visible::<T>(index);

        if is_host_visible {
            self.copy_data_to_buffer::<T, D>(index, dst_byte_offset, data);
        } else {
            self.upload_to_buffer_via_staging::<M, T, D>(index, data, dst_byte_offset);
        }
    }

    /// Upload data to a per-frame buffer for the current frame.
    /// - HOST_VISIBLE: Direct memory copy
    /// - DEVICE_LOCAL: Creates internal staging buffer, transfers data via GPU, then destroys staging
    ///
    /// Offset is always 0.
    pub fn upload_to_per_frame_buffer<
        M: GpuMutex + 'static,
        T: BufferMarker<Lifetime = PerFrame>,
        D,
    >(
        &mut self,
        index: usize,
        data: &[D],
    ) {
        let buffer_size = self
            .get_per_frame_buffer_size::<T>(self.current_frame, index)
            .expect("Failed to get per-frame buffer size");
        let data = clamp_slice_to_buffer_capacity(data, buffer_size);

        let is_host_visible = self.is_per_frame_buffer_host_visible::<T>(self.current_frame, index);

        if is_host_visible {
            self.copy_data_to_per_frame_buffer::<T, D>(self.current_frame, index, data);
        } else {
            self.upload_to_per_frame_buffer_via_staging::<M, T, D>(index, data);
        }
    }

    /// Internal: Upload data to a DEVICE_LOCAL buffer via a temporary staging buffer.
    fn upload_to_buffer_via_staging<
        M: GpuMutex + 'static,
        T: BufferMarker<Lifetime = Persistent>,
        D,
    >(
        &mut self,
        index: usize,
        data: &[D],
        dst_byte_offset: u64,
    ) {
        let data_size = std::mem::size_of_val(data) as u64;
        let queue_family_indices = self.get_queue_family_indices();
        let current_frame = self.current_frame;
        let allocation = self.graphics_staging_buffers[current_frame].allocate_and_write(
            data,
            &self.context.device,
            self.context.physical_device,
            &self.context.instance,
            &queue_family_indices,
        );

        let src = allocation.buffer;
        let dst = self.get_buffer::<T>(index).unwrap();
        let src_size = data_size;
        let dst_size = self.get_buffer_size::<T>(index).unwrap();
        let max_copy_size = dst_size.saturating_sub(dst_byte_offset);
        let size = src_size.min(max_copy_size);
        if size == 0 {
            return;
        }
        let cb = self.command_resource_manager.get_free_command_buffer(
            self.command_resources.graphics_transfer_pool_ids[current_frame],
            vk::CommandBufferLevel::PRIMARY,
        );
        begin_transfer_command_buffer(&self.context.device, cb);
        crate::vulkan::transfer_commands::copy_buffer(
            &self.context.device,
            cb,
            src,
            allocation.offset,
            dst,
            dst_byte_offset,
            size,
        );
        let timeline = if M::IS_NOOP {
            None
        } else {
            Some(self.acquire_mutex(TypeId::of::<M>()))
        };
        let completion_value = self.graphics_transfer_next_timeline_value;
        self.graphics_transfer_next_timeline_value += 1;
        self.graphics_transfer_last_submitted_values[current_frame] = completion_value;
        submit_transfer_command_buffer(
            &self.context.device,
            self.context.queues.transfer,
            cb,
            timeline.as_slice(),
            Some((self.sync.graphics_transfer_timeline, completion_value)),
        );
    }

    /// Internal: Upload data to a DEVICE_LOCAL per-frame buffer via a temporary staging buffer.
    fn upload_to_per_frame_buffer_via_staging<
        M: GpuMutex + 'static,
        T: BufferMarker<Lifetime = PerFrame>,
        D,
    >(
        &mut self,
        index: usize,
        data: &[D],
    ) {
        let data_size = std::mem::size_of_val(data) as u64;
        let queue_family_indices = self.get_queue_family_indices();
        let current_frame = self.current_frame;
        let allocation = self.graphics_staging_buffers[current_frame].allocate_and_write(
            data,
            &self.context.device,
            self.context.physical_device,
            &self.context.instance,
            &queue_family_indices,
        );

        let src = allocation.buffer;
        let dst = self
            .get_per_frame_buffer::<T>(current_frame, index)
            .unwrap();
        let src_size = data_size;
        let dst_size = self
            .get_per_frame_buffer_size::<T>(current_frame, index)
            .unwrap();
        let size = src_size.min(dst_size);
        let cb = self.command_resource_manager.get_free_command_buffer(
            self.command_resources.graphics_transfer_pool_ids[current_frame],
            vk::CommandBufferLevel::PRIMARY,
        );
        begin_transfer_command_buffer(&self.context.device, cb);
        crate::vulkan::transfer_commands::copy_buffer(
            &self.context.device,
            cb,
            src,
            allocation.offset,
            dst,
            0,
            size,
        );
        let timeline = if M::IS_NOOP {
            None
        } else {
            Some(self.acquire_mutex(TypeId::of::<M>()))
        };
        let completion_value = self.graphics_transfer_next_timeline_value;
        self.graphics_transfer_next_timeline_value += 1;
        self.graphics_transfer_last_submitted_values[current_frame] = completion_value;
        submit_transfer_command_buffer(
            &self.context.device,
            self.context.queues.transfer,
            cb,
            timeline.as_slice(),
            Some((self.sync.graphics_transfer_timeline, completion_value)),
        );
    }

    /// Upload data to an image resource.
    ///
    /// This method:
    /// 1. Creates a temporary staging buffer internally
    /// 2. Copies data to the staging buffer
    /// 3. Executes GPU transfer to the image (synchronous)
    /// 4. Destroys the staging buffer
    ///
    /// Users do not need to specify the staging buffer type - it's managed internally.
    ///
    /// # Arguments
    /// * `image_index` - Index of the image in the marker type's image array
    /// * `data` - Raw image data (e.g., RGBA8 format)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `depth` - Image depth in layers (1 for 2D images)
    pub fn upload_to_image<M: GpuMutex + 'static, T: ImageMarker<Lifetime = Persistent>>(
        &mut self,
        image_index: usize,
        data: &[u8],
        width: u32,
        height: u32,
        depth: u32,
    ) {
        debug_assert!(width >= 1, "upload_to_image: width must be >= 1");
        debug_assert!(height >= 1, "upload_to_image: height must be >= 1");
        debug_assert!(depth >= 1, "upload_to_image: depth must be >= 1");

        let dst_image = self.get_image::<T>(image_index).unwrap();
        let extent = vk::Extent3D {
            width,
            height,
            depth,
        };

        let expected_image_size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(depth as usize)
            .saturating_mul(4);
        let owned_upload_data;
        let upload_data = if data.len() < expected_image_size {
            owned_upload_data = {
                let mut padded = vec![0u8; expected_image_size];
                padded[..data.len()].copy_from_slice(data);
                padded
            };
            &owned_upload_data[..]
        } else if data.len() > expected_image_size {
            &data[..expected_image_size]
        } else {
            data
        };

        let queue_family_indices = self.get_queue_family_indices();
        let current_frame = self.current_frame;
        let allocation = self.graphics_staging_buffers[current_frame].allocate_and_write(
            upload_data,
            &self.context.device,
            self.context.physical_device,
            &self.context.instance,
            &queue_family_indices,
        );

        let src_buffer = allocation.buffer;
        let cb = self.command_resource_manager.get_free_command_buffer(
            self.command_resources.graphics_transfer_pool_ids[current_frame],
            vk::CommandBufferLevel::PRIMARY,
        );
        begin_transfer_command_buffer(&self.context.device, cb);
        crate::vulkan::transfer_commands::copy_buffer_to_image(
            &self.context.device,
            cb,
            src_buffer,
            allocation.offset,
            dst_image,
            extent,
        );
        let timeline = if M::IS_NOOP {
            None
        } else {
            Some(self.acquire_mutex(TypeId::of::<M>()))
        };
        let completion_value = self.graphics_transfer_next_timeline_value;
        self.graphics_transfer_next_timeline_value += 1;
        self.graphics_transfer_last_submitted_values[current_frame] = completion_value;
        submit_transfer_command_buffer(
            &self.context.device,
            self.context.queues.transfer,
            cb,
            timeline.as_slice(),
            Some((self.sync.graphics_transfer_timeline, completion_value)),
        );
    }

    /// Transition image layout.
    ///
    /// Executes image layout transition using the transfer queue and command pool.
    ///
    /// # Arguments
    /// * `index` - Index of the image in the marker type's image array
    /// * `old_layout` - Current image layout
    /// * `new_layout` - Target image layout
    /// * `aspect_mask` - Image aspect flags (e.g., COLOR, DEPTH, STENCIL)
    pub fn transition_image_layout<M: GpuMutex + 'static, T: ImageMarker<Lifetime = Persistent>>(
        &mut self,
        index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        let image = self.get_image::<T>(index).unwrap();
        let cb = self.command_resource_manager.get_free_command_buffer(
            self.command_resources.graphics_transfer_pool_ids[self.current_frame],
            vk::CommandBufferLevel::PRIMARY,
        );
        begin_transfer_command_buffer(&self.context.device, cb);
        crate::vulkan::transfer_commands::transition_image_layout(
            &self.context.device,
            cb,
            image,
            old_layout,
            new_layout,
            aspect_mask,
        );
        let timeline = if M::IS_NOOP {
            None
        } else {
            Some(self.acquire_mutex(TypeId::of::<M>()))
        };
        let completion_value = self.graphics_transfer_next_timeline_value;
        self.graphics_transfer_next_timeline_value += 1;
        self.graphics_transfer_last_submitted_values[self.current_frame] = completion_value;
        submit_transfer_command_buffer(
            &self.context.device,
            self.context.queues.transfer,
            cb,
            timeline.as_slice(),
            Some((self.sync.graphics_transfer_timeline, completion_value)),
        );
    }

    /// Transition per-frame image layout.
    ///
    /// Executes image layout transition for a specific frame using the graphics queue and command pool.
    ///
    /// # Arguments
    /// * `frame_index` - Frame index (0 to MAX_FRAMES_IN_FLIGHT-1)
    /// * `image_index` - Index of the image in the marker type's per-frame image array
    /// * `old_layout` - Current image layout
    /// * `new_layout` - Target image layout
    /// * `aspect_mask` - Image aspect flags (e.g., COLOR, DEPTH, STENCIL)
    pub fn transition_per_frame_image_layout<
        M: GpuMutex + 'static,
        T: ImageMarker<Lifetime = PerFrame>,
    >(
        &mut self,
        frame_index: usize,
        image_index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        let image = self
            .get_per_frame_image::<T>(frame_index, image_index)
            .unwrap();
        let cb = self.command_resource_manager.get_free_command_buffer(
            self.command_resources.graphics_transfer_pool_ids[self.current_frame],
            vk::CommandBufferLevel::PRIMARY,
        );
        begin_transfer_command_buffer(&self.context.device, cb);
        crate::vulkan::transfer_commands::transition_image_layout(
            &self.context.device,
            cb,
            image,
            old_layout,
            new_layout,
            aspect_mask,
        );
        let timeline = if M::IS_NOOP {
            None
        } else {
            Some(self.acquire_mutex(TypeId::of::<M>()))
        };
        let completion_value = self.graphics_transfer_next_timeline_value;
        self.graphics_transfer_next_timeline_value += 1;
        self.graphics_transfer_last_submitted_values[self.current_frame] = completion_value;
        submit_transfer_command_buffer(
            &self.context.device,
            self.context.queues.transfer,
            cb,
            timeline.as_slice(),
            Some((self.sync.graphics_transfer_timeline, completion_value)),
        );
    }

    pub fn transition_per_frame_image_layouts<
        M: GpuMutex + 'static,
        T: ImageMarker<Lifetime = PerFrame>,
    >(
        &mut self,
        target_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        for frame_index in 0..MAX_FRAMES_IN_FLIGHT {
            self.transition_per_frame_image_layout::<M, T>(
                frame_index,
                0,
                vk::ImageLayout::UNDEFINED,
                target_layout,
                aspect_mask,
            );
        }
    }

    // ========================================================================
    // Bind APIs - Type-based descriptor set binding
    // ========================================================================

    /// Create a sampler and register it with the given marker type.
    pub fn create_sampler<T: SamplerMarker>(&mut self) {
        let atomic_id = T::id();

        let old_id = SamplerId::load(atomic_id);
        if old_id != SamplerId::INVALID {
            self.resource_manager
                .write()
                .unwrap()
                .destroy_sampler(old_id);
            SamplerId::INVALID.store(atomic_id);
        }

        let new_id = self
            .resource_manager
            .write()
            .unwrap()
            .create_sampler(&T::CONFIG);
        new_id.store(atomic_id);
    }

    /// Bind regular buffers to a descriptor set as a descriptor array (same buffers for all frames).
    ///
    /// # Arguments
    /// * `binding` - The binding number in the descriptor set layout
    /// * `array_index` - Starting index in the descriptor array
    /// * `buffer_index` - Starting index in the marker type's buffer array
    /// * `count` - Number of buffers to bind
    /// * `descriptor_type` - The type of descriptor (UNIFORM_BUFFER or STORAGE_BUFFER)
    pub fn bind_buffer<
        D: DescriptorSetMarker<Lifetime = Persistent> + 'static,
        B: BufferMarker<Lifetime = Persistent> + 'static,
    >(
        &mut self,
        binding: u32,
        array_index: u32,
        buffer_index: usize,
        count: usize,
        descriptor_type: vk::DescriptorType,
    ) {
        let set_id = DescriptorSetId::load(&D::ids_slice()[0]);
        let mut buffer_infos = Vec::with_capacity(count);
        for i in 0..count {
            let buffer = self
                .get_buffer::<B>(buffer_index + i)
                .expect("Failed to get buffer for binding");
            let buffer_size = self
                .get_buffer_size::<B>(buffer_index + i)
                .expect("Failed to get buffer size for binding");
            buffer_infos.push(vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: buffer_size,
            });
        }

        self.descriptor_manager.write().unwrap().update_buffer(
            set_id,
            binding,
            array_index,
            &buffer_infos,
            descriptor_type,
        );
    }

    /// Bind per-frame buffers to a descriptor set as a descriptor array.
    ///
    /// # Arguments
    /// * `binding` - The binding number in the descriptor set layout
    /// * `array_index` - Starting index in the descriptor array
    /// * `buffer_index` - Starting index in the marker type's buffer array
    /// * `count` - Number of buffers to bind
    /// * `descriptor_type` - The type of descriptor (UNIFORM_BUFFER or STORAGE_BUFFER)
    pub fn bind_per_frame_buffer<
        D: DescriptorSetMarker<Lifetime = PerFrame> + 'static,
        B: BufferMarker<Lifetime = PerFrame> + 'static,
    >(
        &mut self,
        binding: u32,
        array_index: u32,
        buffer_index: usize,
        count: usize,
        descriptor_type: vk::DescriptorType,
    ) {
        let frame = self.current_frame;
        let set_id = DescriptorSetId::load(&D::ids_slice()[frame]);

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = (0..count)
            .map(|i| {
                let buffer = self
                    .get_per_frame_buffer::<B>(frame, buffer_index + i)
                    .expect("Failed to get per-frame buffer for binding");
                let buffer_size = self
                    .get_per_frame_buffer_size::<B>(frame, buffer_index + i)
                    .expect("Failed to get per-frame buffer size for binding");
                vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: buffer_size,
                }
            })
            .collect();

        self.descriptor_manager.write().unwrap().update_buffer(
            set_id,
            binding,
            array_index,
            &buffer_infos,
            descriptor_type,
        );
    }

    /// Bind regular images to a descriptor set as a descriptor array (same images for all frames).
    ///
    /// # Arguments
    /// * `binding` - The binding number in the descriptor set layout
    /// * `array_index` - Starting index in the descriptor array
    /// * `image_index` - Starting index in the marker type's image array
    /// * `count` - Number of images to bind
    pub fn bind_image<
        D: DescriptorSetMarker<Lifetime = Persistent> + 'static,
        V: ImageViewMarker<Lifetime = Persistent>,
        S: SamplerMarker,
    >(
        &mut self,
        binding: u32,
        array_index: u32,
        image_index: usize,
        count: usize,
    ) {
        let set_id = DescriptorSetId::load(&D::ids_slice()[0]);
        let sampler = self
            .get_sampler::<S>()
            .expect("Failed to get sampler for binding");

        let mut image_infos = Vec::with_capacity(count);
        for i in 0..count {
            let image_view = self
                .get_image_view::<V>(image_index + i)
                .expect("Failed to get image view for binding");
            image_infos.push(vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image_view,
                sampler,
            });
        }

        self.descriptor_manager.write().unwrap().update_image(
            set_id,
            binding,
            array_index,
            &image_infos,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
    }

    /// Bind per-frame images to a descriptor set as a descriptor array.
    ///
    /// # Arguments
    /// * `binding` - The binding number in the descriptor set layout
    /// * `array_index` - Starting index in the descriptor array
    /// * `image_index` - Starting index in the marker type's image array
    /// * `count` - Number of images to bind
    pub fn bind_per_frame_image<
        D: DescriptorSetMarker<Lifetime = PerFrame> + 'static,
        V: ImageViewMarker<Lifetime = PerFrame>,
        S: SamplerMarker,
    >(
        &mut self,
        binding: u32,
        array_index: u32,
        image_index: usize,
        count: usize,
    ) {
        let frame_index = self.current_frame;
        let set_id = DescriptorSetId::load(&D::ids_slice()[frame_index]);
        let sampler = self
            .get_sampler::<S>()
            .expect("Failed to get sampler for binding");

        let image_infos: Vec<vk::DescriptorImageInfo> = (0..count)
            .map(|i| {
                let image_view = self
                    .get_per_frame_image_view::<V>(frame_index, image_index + i)
                    .expect("Failed to get per-frame image view for binding");
                vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image_view,
                    sampler,
                }
            })
            .collect();

        self.descriptor_manager.write().unwrap().update_image(
            set_id,
            binding,
            array_index,
            &image_infos,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
    }

    // ========================================================================
    // Other APIs
    // ========================================================================

    pub(crate) fn set_frame_submit_groups(
        &mut self,
        submit_groups: Vec<Box<dyn ErasedGraphicsSubmitGroup>>,
    ) {
        self.frame_submit_groups = submit_groups;
    }

    pub fn acquire_mutex_by_type<M: GpuMutex>(&mut self) -> (vk::Semaphore, u64, u64) {
        self.acquire_mutex(TypeId::of::<M>())
    }
}

pub(crate) fn acquire_mutex_state(
    gpu_mutexes: &mut HashMap<TypeId, GpuMutexState>,
    device: &ash::Device,
    type_id: TypeId,
) -> (vk::Semaphore, u64, u64) {
    let state = gpu_mutexes
        .entry(type_id)
        .or_insert_with(|| GpuMutexState::new(device));
    let (wait_value, signal_value) = state.acquire();
    (state.semaphore, wait_value, signal_value)
}

pub(super) fn clamp_slice_to_buffer_capacity<D>(data: &[D], buffer_size: u64) -> &[D] {
    let data_size = std::mem::size_of_val(data);
    if data_size <= buffer_size as usize {
        return data;
    }

    let element_size = std::mem::size_of::<D>();
    let max_elements = if element_size == 0 {
        data.len()
    } else {
        (buffer_size as usize) / element_size
    };
    eprintln!(
        "Warning: Buffer overflow prevented - truncated {} elements to {} (buffer capacity)",
        data.len(),
        max_elements
    );
    &data[..max_elements]
}

pub(crate) fn begin_transfer_command_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
) {
    let begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        p_inheritance_info: ptr::null(),
        _marker: PhantomData,
    };

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("Failed to begin transfer command buffer");
    }
}

pub(crate) fn submit_transfer_command_buffer(
    device: &ash::Device,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
    timelines: &[(vk::Semaphore, u64, u64)],
    completion_timeline: Option<(vk::Semaphore, u64)>,
) {
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("Failed to end transfer command buffer");
    }

    let count = timelines.len();
    let mut wait_semaphores = Vec::with_capacity(count);
    let mut wait_values = Vec::with_capacity(count);
    let mut signal_semaphores =
        Vec::with_capacity(count + usize::from(completion_timeline.is_some()));
    let mut signal_values = Vec::with_capacity(count + usize::from(completion_timeline.is_some()));
    let mut wait_stage_masks = Vec::with_capacity(count);

    for &(semaphore, wait_value, signal_value) in timelines {
        wait_semaphores.push(semaphore);
        wait_values.push(wait_value);
        signal_semaphores.push(semaphore);
        signal_values.push(signal_value);
        wait_stage_masks.push(vk::PipelineStageFlags::TRANSFER);
    }

    if let Some((semaphore, value)) = completion_timeline {
        signal_semaphores.push(semaphore);
        signal_values.push(value);
    }

    let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
        s_type: vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_value_count: wait_values.len() as u32,
        p_wait_semaphore_values: wait_values.as_ptr(),
        signal_semaphore_value_count: signal_values.len() as u32,
        p_signal_semaphore_values: signal_values.as_ptr(),
        _marker: PhantomData,
    };

    let p_next = if wait_values.is_empty() && signal_values.is_empty() {
        ptr::null()
    } else {
        (&timeline_submit_info as *const vk::TimelineSemaphoreSubmitInfo).cast()
    };

    let submit_info = vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        p_next,
        wait_semaphore_count: count as u32,
        p_wait_semaphores: wait_semaphores.as_ptr(),
        p_wait_dst_stage_mask: wait_stage_masks.as_ptr(),
        command_buffer_count: 1,
        p_command_buffers: &command_buffer,
        signal_semaphore_count: signal_semaphores.len() as u32,
        p_signal_semaphores: signal_semaphores.as_ptr(),
        _marker: PhantomData,
    };

    unsafe {
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .expect("Failed to submit transfer command buffer");
    }
}

pub(crate) fn wait_for_timeline_value(device: &ash::Device, semaphore: vk::Semaphore, value: u64) {
    if value == 0 {
        return;
    }

    let wait_info = vk::SemaphoreWaitInfo {
        s_type: vk::StructureType::SEMAPHORE_WAIT_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreWaitFlags::empty(),
        semaphore_count: 1,
        p_semaphores: &semaphore,
        p_values: &value,
        _marker: PhantomData,
    };

    unsafe {
        device
            .wait_semaphores(&wait_info, u64::MAX)
            .expect("Failed to wait for timeline semaphore");
    }
}

impl VulkanRendererCore {
    /// Clean up gpu_mutexes owned by core.
    pub(super) fn cleanup(&mut self) {
        let mut gpu_mutexes = self.gpu_mutexes.lock().unwrap();
        unsafe {
            for state in gpu_mutexes.values() {
                self.context.device.destroy_semaphore(state.semaphore, None);
            }
        }
        gpu_mutexes.clear();
    }
}

impl Drop for VulkanRendererCore {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .device_wait_idle()
                .expect("Failed to wait for device idle");

            #[cfg(feature = "gpu-profiling")]
            self.gpu_timing.destroy(&self.context.device);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.context
                    .device
                    .destroy_semaphore(self.sync.image_available_semaphores[i], None);
                self.context
                    .device
                    .destroy_fence(self.sync.in_flight_fences[i], None);
            }
            for semaphore in &self.sync.render_finished_semaphores {
                self.context.device.destroy_semaphore(*semaphore, None);
            }
            self.context
                .device
                .destroy_semaphore(self.sync.graphics_transfer_timeline, None);

            // Explicitly clean up command resource manager (command pools and buffers)
            self.command_resource_manager.cleanup();

            self.swapchain.cleanup(&self.context.device);

            for staging_buffer in &mut self.graphics_staging_buffers {
                staging_buffer.destroy(&self.context.device);
            }

            // Explicitly clean up resource manager (buffers, images, image views, samplers)
            self.resource_manager.write().unwrap().cleanup();

            // Explicitly clean up descriptor resources before destroying device
            self.descriptor_manager.write().unwrap().cleanup();

            // Explicitly clean up pipelines and layouts before destroying device
            self.pipeline_manager.write().unwrap().cleanup();

            self.context.device.destroy_device(None);

            self.context
                .surface_loader
                .destroy_surface(self.context.surface, None);

            if VALIDATION.is_enable {
                self.context
                    .debug_utils_loader
                    .destroy_debug_utils_messenger(self.context.debug_messenger, None);
            }
            self.context.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::clamp_slice_to_buffer_capacity;

    #[test]
    fn clamp_slice_to_buffer_capacity_keeps_data_when_it_fits() {
        let data = [1u32, 2, 3, 4];
        let clamped =
            clamp_slice_to_buffer_capacity(&data, (std::mem::size_of::<u32>() * 4) as u64);

        assert_eq!(clamped.len(), 4);
        assert_eq!(clamped, &data);
    }

    #[test]
    fn clamp_slice_to_buffer_capacity_truncates_when_buffer_is_smaller() {
        let data = [1u32, 2, 3, 4];
        let clamped =
            clamp_slice_to_buffer_capacity(&data, (std::mem::size_of::<u32>() * 2) as u64);

        assert_eq!(clamped.len(), 2);
        assert_eq!(clamped, &[1u32, 2]);
    }
}
