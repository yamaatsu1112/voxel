use ash::vk;

/// Configuration for creating a buffer.
#[derive(Debug, Clone, Copy)]
pub struct BufferConfig {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

/// Size format for images, supporting both swapchain-relative and fixed sizes.
#[derive(Debug, Clone, Copy)]
pub enum ImageSizeFormat {
    /// Size matches the swapchain extent
    Swapchain,
    /// Fixed size in pixels
    Fixed { width: u32, height: u32 },
    /// Fixed 3D size in pixels
    Fixed3D { width: u32, height: u32, depth: u32 },
    /// Dynamic size that must be specified at creation time.
    Dynamic,
}

/// Configuration for creating an image.
#[derive(Debug, Clone, Copy)]
pub struct ImageConfig {
    pub size: ImageSizeFormat,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

/// Configuration for creating an image view.
#[derive(Debug, Clone, Copy)]
pub struct ImageViewConfig {
    pub format: vk::Format,
    pub aspect_mask: vk::ImageAspectFlags,
    pub view_type: vk::ImageViewType,
}

/// Configuration for creating a sampler.
#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub address_mode: vk::SamplerAddressMode,
    pub mipmap_mode: vk::SamplerMipmapMode,
}

/// Configuration for creating a descriptor set layout.
pub struct DescriptorSetConfig {
    pub layout_bindings: &'static [vk::DescriptorSetLayoutBinding<'static>],
}
