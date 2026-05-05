use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::vulkan::resource_config::{
    BufferConfig, BufferId, ImageConfig, ImageId, ImageSizeFormat, ImageViewConfig, ImageViewId,
    SamplerConfig, SamplerId,
};

struct BufferInfo {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    _usage: vk::BufferUsageFlags,
    mapped_ptr: Option<*mut u8>,
}

struct ImageInfo {
    image: vk::Image,
    memory: vk::DeviceMemory,
    _format: vk::Format,
    extent: vk::Extent3D,
    _usage: vk::ImageUsageFlags,
    _tiling: vk::ImageTiling,
}

pub struct ResourceManager {
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    buffers: Vec<Option<BufferInfo>>,
    free_buffer_indices: Vec<usize>,
    images: Vec<Option<ImageInfo>>,
    free_image_indices: Vec<usize>,
    image_views: Vec<Option<vk::ImageView>>,
    free_image_view_indices: Vec<usize>,
    samplers: Vec<Option<vk::Sampler>>,
    free_sampler_indices: Vec<usize>,
}

impl ResourceManager {
    pub fn new(
        device: ash::Device,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        Self {
            device,
            instance,
            physical_device,
            buffers: Vec::new(),
            free_buffer_indices: Vec::new(),
            images: Vec::new(),
            free_image_indices: Vec::new(),
            image_views: Vec::new(),
            free_image_view_indices: Vec::new(),
            samplers: Vec::new(),
            free_sampler_indices: Vec::new(),
        }
    }

    fn allocate_buffer_slot(&mut self, info: BufferInfo) -> BufferId {
        if let Some(index) = self.free_buffer_indices.pop() {
            self.buffers[index] = Some(info);
            BufferId(index)
        } else {
            let index = self.buffers.len();
            self.buffers.push(Some(info));
            BufferId(index)
        }
    }

    fn allocate_image_slot(&mut self, info: ImageInfo) -> ImageId {
        if let Some(index) = self.free_image_indices.pop() {
            self.images[index] = Some(info);
            ImageId(index)
        } else {
            let index = self.images.len();
            self.images.push(Some(info));
            ImageId(index)
        }
    }

    fn allocate_image_view_slot(&mut self, view: vk::ImageView) -> ImageViewId {
        if let Some(index) = self.free_image_view_indices.pop() {
            self.image_views[index] = Some(view);
            ImageViewId(index)
        } else {
            let index = self.image_views.len();
            self.image_views.push(Some(view));
            ImageViewId(index)
        }
    }

    fn allocate_sampler_slot(&mut self, sampler: vk::Sampler) -> SamplerId {
        if let Some(index) = self.free_sampler_indices.pop() {
            self.samplers[index] = Some(sampler);
            SamplerId(index)
        } else {
            let index = self.samplers.len();
            self.samplers.push(Some(sampler));
            SamplerId(index)
        }
    }

    pub fn create_buffer(
        &mut self,
        config: &BufferConfig,
        queue_family_indices: &[u32],
    ) -> BufferId {
        let buffer_info = self.create_buffer_internal(config, queue_family_indices);
        self.allocate_buffer_slot(buffer_info)
    }

    pub fn destroy_buffer(&mut self, id: BufferId) {
        if id == BufferId::INVALID {
            return;
        }
        if let Some(slot) = self.buffers.get_mut(id.0)
            && let Some(info) = slot.take()
        {
            self.destroy_buffer_internal(info);
            self.free_buffer_indices.push(id.0);
        }
    }

    pub fn get_buffer(&self, id: BufferId) -> Option<vk::Buffer> {
        self.buffers
            .get(id.0)
            .and_then(|opt| opt.as_ref())
            .map(|info| info.buffer)
    }

    pub fn get_buffer_mapped_ptr(&self, id: BufferId) -> Option<*mut u8> {
        self.buffers
            .get(id.0)
            .and_then(|opt| opt.as_ref())
            .and_then(|info| info.mapped_ptr)
    }

    pub fn get_buffer_size(&self, id: BufferId) -> Option<u64> {
        self.buffers
            .get(id.0)
            .and_then(|opt| opt.as_ref())
            .map(|info| info.size)
    }

    pub fn is_buffer_host_visible(&self, id: BufferId) -> bool {
        self.buffers
            .get(id.0)
            .and_then(|opt| opt.as_ref())
            .map(|info| info.mapped_ptr.is_some())
            .unwrap_or(false)
    }

    pub fn copy_data_to_buffer<D>(&self, id: BufferId, dst_byte_offset: u64, data: &[D]) {
        let mapped_ptr = self
            .get_buffer_mapped_ptr(id)
            .expect("Failed to get mapped pointer for buffer");
        let data_size = std::mem::size_of_val(data);
        if data_size == 0 {
            return;
        }
        let buffer_size = self
            .get_buffer_size(id)
            .expect("Failed to get buffer size for copy");
        let max_copy_size = buffer_size.saturating_sub(dst_byte_offset);
        let copy_size = data_size.min(max_copy_size as usize);
        if copy_size == 0 {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped_ptr.add(dst_byte_offset as usize),
                copy_size,
            );
        }
    }

    pub fn read_buffer_data<D: Copy, const N: usize>(&self, id: BufferId) -> Option<[D; N]> {
        let mapped_ptr = self.get_buffer_mapped_ptr(id)?;
        unsafe {
            let mut data = std::mem::MaybeUninit::<[D; N]>::uninit();
            std::ptr::copy_nonoverlapping(mapped_ptr as *const D, data.as_mut_ptr() as *mut D, N);
            Some(data.assume_init())
        }
    }

    pub fn create_image(
        &mut self,
        config: &ImageConfig,
        swapchain_extent: vk::Extent2D,
        queue_family_indices: &[u32],
    ) -> ImageId {
        let (width, height, depth) = match config.size {
            ImageSizeFormat::Swapchain => (swapchain_extent.width, swapchain_extent.height, 1),
            ImageSizeFormat::Fixed { width, height } => (width, height, 1),
            ImageSizeFormat::Fixed3D {
                width,
                height,
                depth,
            } => (width, height, depth),
            ImageSizeFormat::Dynamic => {
                panic!(
                    "ImageSizeFormat::Dynamic must be resolved before ResourceManager::create_image"
                )
            }
        };

        let image_info = self.create_image_internal(
            width,
            height,
            depth,
            config.format,
            config.usage,
            config.properties,
            queue_family_indices,
        );
        self.allocate_image_slot(image_info)
    }

    pub fn destroy_image(&mut self, id: ImageId) {
        if id == ImageId::INVALID {
            return;
        }
        if let Some(slot) = self.images.get_mut(id.0)
            && let Some(info) = slot.take()
        {
            self.destroy_image_internal(info);
            self.free_image_indices.push(id.0);
        }
    }

    pub fn get_image(&self, id: ImageId) -> Option<vk::Image> {
        self.images
            .get(id.0)
            .and_then(|opt| opt.as_ref())
            .map(|info| info.image)
    }

    pub fn get_image_extent(&self, id: ImageId) -> Option<vk::Extent3D> {
        self.images
            .get(id.0)
            .and_then(|opt| opt.as_ref())
            .map(|info| info.extent)
    }

    pub fn create_image_view(&mut self, config: &ImageViewConfig, image: vk::Image) -> ImageViewId {
        let image_view = self.create_image_view_internal(
            image,
            config.format,
            config.aspect_mask,
            config.view_type,
        );
        self.allocate_image_view_slot(image_view)
    }

    pub fn destroy_image_view(&mut self, id: ImageViewId) {
        if id == ImageViewId::INVALID {
            return;
        }
        if let Some(slot) = self.image_views.get_mut(id.0)
            && let Some(view) = slot.take()
        {
            unsafe {
                self.device.destroy_image_view(view, None);
            }
            self.free_image_view_indices.push(id.0);
        }
    }

    pub fn get_image_view(&self, id: ImageViewId) -> Option<vk::ImageView> {
        self.image_views
            .get(id.0)
            .and_then(|opt| opt.as_ref())
            .copied()
    }

    pub fn create_sampler(&mut self, config: &SamplerConfig) -> SamplerId {
        let sampler_info = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: config.mag_filter,
            min_filter: config.min_filter,
            address_mode_u: config.address_mode,
            address_mode_v: config.address_mode,
            address_mode_w: config.address_mode,
            anisotropy_enable: vk::FALSE,
            max_anisotropy: 1.0,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            mip_lod_bias: 0.0,
            mipmap_mode: config.mipmap_mode,
            min_lod: 0.0,
            max_lod: 1.0,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            unnormalized_coordinates: vk::FALSE,
            _marker: PhantomData,
        };

        let sampler = unsafe {
            self.device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create sampler")
        };

        self.allocate_sampler_slot(sampler)
    }

    pub fn destroy_sampler(&mut self, id: SamplerId) {
        if id == SamplerId::INVALID {
            return;
        }
        if let Some(slot) = self.samplers.get_mut(id.0)
            && let Some(sampler) = slot.take()
        {
            unsafe {
                self.device.destroy_sampler(sampler, None);
            }
            self.free_sampler_indices.push(id.0);
        }
    }

    pub fn get_sampler(&self, id: SamplerId) -> Option<vk::Sampler> {
        if id == SamplerId::INVALID {
            return None;
        }
        self.samplers.get(id.0).and_then(|opt| *opt)
    }

    pub fn cleanup(&mut self) {
        unsafe {
            for view in self.image_views.drain(..).flatten() {
                self.device.destroy_image_view(view, None);
            }
            self.free_image_view_indices.clear();

            for image_info in self.images.drain(..).flatten() {
                self.device.destroy_image(image_info.image, None);
                self.device.free_memory(image_info.memory, None);
            }
            self.free_image_indices.clear();

            for buffer_info in self.buffers.drain(..).flatten() {
                if let Some(mapped_ptr) = buffer_info.mapped_ptr
                    && !mapped_ptr.is_null()
                {
                    self.device.unmap_memory(buffer_info.memory);
                }
                self.device.destroy_buffer(buffer_info.buffer, None);
                self.device.free_memory(buffer_info.memory, None);
            }
            self.free_buffer_indices.clear();

            for sampler in self.samplers.drain(..).flatten() {
                self.device.destroy_sampler(sampler, None);
            }
            self.free_sampler_indices.clear();
        }
    }

    fn create_buffer_internal(
        &self,
        config: &BufferConfig,
        queue_family_indices: &[u32],
    ) -> BufferInfo {
        let (sharing_mode, queue_family_index_count, queue_family_indices_ptr) =
            if queue_family_indices.len() > 1 {
                (
                    vk::SharingMode::CONCURRENT,
                    queue_family_indices.len() as u32,
                    queue_family_indices.as_ptr(),
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, ptr::null())
            };

        let create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size: config.size,
            usage: config.usage,
            sharing_mode,
            queue_family_index_count,
            p_queue_family_indices: queue_family_indices_ptr,
            ..Default::default()
        };

        let buffer = unsafe {
            self.device
                .create_buffer(&create_info, None)
                .expect("Failed to create buffer")
        };

        let memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let memory_type_index = Self::find_memory_type(
            &self.instance,
            self.physical_device,
            memory_requirements.memory_type_bits,
            config.properties,
        );

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: memory_requirements.size,
            memory_type_index,
            ..Default::default()
        };

        let buffer_memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate buffer memory")
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind buffer memory");
        }

        let mapped_ptr = if config
            .properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            unsafe {
                self.device
                    .map_memory(buffer_memory, 0, config.size, vk::MemoryMapFlags::empty())
                    .ok()
                    .map(|ptr| ptr as *mut u8)
            }
        } else {
            None
        };

        BufferInfo {
            buffer,
            memory: buffer_memory,
            size: config.size,
            _usage: config.usage,
            mapped_ptr,
        }
    }

    fn destroy_buffer_internal(&self, buffer_info: BufferInfo) {
        unsafe {
            if let Some(mapped_ptr) = buffer_info.mapped_ptr
                && !mapped_ptr.is_null()
            {
                self.device.unmap_memory(buffer_info.memory);
            }
            self.device.destroy_buffer(buffer_info.buffer, None);
            self.device.free_memory(buffer_info.memory, None);
        }
    }

    fn create_image_internal(
        &self,
        width: u32,
        height: u32,
        depth: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
        queue_family_indices: &[u32],
    ) -> ImageInfo {
        let extent = vk::Extent3D {
            width,
            height,
            depth,
        };

        let image_type = if depth > 1 {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        };

        let (sharing_mode, queue_family_index_count, queue_family_indices_ptr) =
            if queue_family_indices.len() > 1 {
                (
                    vk::SharingMode::CONCURRENT,
                    queue_family_indices.len() as u32,
                    queue_family_indices.as_ptr(),
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, ptr::null())
            };

        let create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type,
            format,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode,
            queue_family_index_count,
            p_queue_family_indices: queue_family_indices_ptr,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image = unsafe {
            self.device
                .create_image(&create_info, None)
                .expect("Failed to create image")
        };

        let memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let memory_type_index = Self::find_memory_type(
            &self.instance,
            self.physical_device,
            memory_requirements.memory_type_bits,
            properties,
        );

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: memory_requirements.size,
            memory_type_index,
            ..Default::default()
        };

        let image_memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate image memory")
        };

        unsafe {
            self.device
                .bind_image_memory(image, image_memory, 0)
                .expect("Failed to bind image memory");
        }

        ImageInfo {
            image,
            memory: image_memory,
            _format: format,
            extent,
            _usage: usage,
            _tiling: vk::ImageTiling::OPTIMAL,
        }
    }

    fn destroy_image_internal(&self, image_info: ImageInfo) {
        unsafe {
            self.device.destroy_image(image_info.image, None);
            self.device.free_memory(image_info.memory, None);
        }
    }

    fn create_image_view_internal(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
    ) -> vk::ImageView {
        let components = vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        };

        let image_view_create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            image,
            view_type,
            format,
            components,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        unsafe {
            self.device
                .create_image_view(&image_view_create_info, None)
                .expect("Failed to create image view")
        }
    }

    fn find_memory_type(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        for i in 0..memory_properties.memory_type_count {
            let memory_type = memory_properties.memory_types[i as usize];
            if (type_filter & (1 << i) != 0)
                && (memory_type.property_flags & properties == properties)
            {
                return i;
            }
        }

        panic!("Failed to find suitable memory type");
    }
}
