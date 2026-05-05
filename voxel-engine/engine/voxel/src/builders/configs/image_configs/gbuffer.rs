use ash::vk;

use crate::vulkan::resource_config::{ImageConfig, ImageSizeFormat, ImageViewConfig};

engine_macro::define_image! {
    /// G-Buffer image for deferred rendering
    pub struct GBufferImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageConfig {
        size: ImageSizeFormat::Swapchain,
        format: vk::Format::R32G32B32A32_UINT,
        usage: vk::ImageUsageFlags::from_raw(
            vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_image_view! {
    /// G-Buffer image view
    pub struct GBufferImageView;
    image = GBufferImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageViewConfig {
        format: vk::Format::R32G32B32A32_UINT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        view_type: vk::ImageViewType::TYPE_2D,
    };
}

engine_macro::define_image! {
    /// G-Buffer depth image
    pub struct GBufferDepthImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageConfig {
        size: ImageSizeFormat::Swapchain,
        format: vk::Format::R32_SFLOAT,
        usage: vk::ImageUsageFlags::from_raw(
            vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_image_view! {
    /// G-Buffer depth image view
    pub struct GBufferDepthImageView;
    image = GBufferDepthImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageViewConfig {
        format: vk::Format::R32_SFLOAT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        view_type: vk::ImageViewType::TYPE_2D,
    };
}

engine_macro::define_image! {
    /// G-Buffer color image for voxel RGB color
    pub struct GBufferColorImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageConfig {
        size: ImageSizeFormat::Swapchain,
        format: vk::Format::R8G8B8A8_UINT,
        usage: vk::ImageUsageFlags::from_raw(
            vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_image_view! {
    /// G-Buffer color image view
    pub struct GBufferColorImageView;
    image = GBufferColorImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageViewConfig {
        format: vk::Format::R8G8B8A8_UINT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        view_type: vk::ImageViewType::TYPE_2D,
    };
}
