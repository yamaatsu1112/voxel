use ash::vk;

use crate::vulkan::resource_config::{ImageConfig, ImageSizeFormat, ImageViewConfig};

engine_macro::define_image! {
    /// Dynamic G-Buffer image
    pub struct DynamicGBufferImage;
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
    /// Dynamic G-Buffer image view
    pub struct DynamicGBufferImageView;
    image = DynamicGBufferImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageViewConfig {
        format: vk::Format::R32G32B32A32_UINT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        view_type: vk::ImageViewType::TYPE_2D,
    };
}

engine_macro::define_image! {
    /// Dynamic G-Buffer depth image
    pub struct DynamicGBufferDepthImage;
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
    /// Dynamic G-Buffer depth image view
    pub struct DynamicGBufferDepthImageView;
    image = DynamicGBufferDepthImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageViewConfig {
        format: vk::Format::R32_SFLOAT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        view_type: vk::ImageViewType::TYPE_2D,
    };
}

engine_macro::define_image! {
    /// Dynamic G-Buffer color image
    pub struct DynamicGBufferColorImage;
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
    /// Dynamic G-Buffer color image view
    pub struct DynamicGBufferColorImageView;
    image = DynamicGBufferColorImage;
    lifetime = PerFrame;
    count = 1;
    config = ImageViewConfig {
        format: vk::Format::R8G8B8A8_UINT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        view_type: vk::ImageViewType::TYPE_2D,
    };
}
