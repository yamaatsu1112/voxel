use ash::vk;

use crate::vulkan::resource_config::{ImageConfig, ImageSizeFormat, ImageViewConfig};

engine_macro::define_image! {
    /// UI texture image
    pub struct UITextureImage;
    lifetime = Persistent;
    count = 1;
    config = ImageConfig {
        size: ImageSizeFormat::Dynamic,
        format: vk::Format::R8G8B8A8_SRGB,
        usage: vk::ImageUsageFlags::from_raw(
            vk::ImageUsageFlags::TRANSFER_DST.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_image_view! {
    /// UI texture image view
    pub struct UITextureImageView;
    image = UITextureImage;
    lifetime = Persistent;
    count = 1;
    config = ImageViewConfig {
        format: vk::Format::R8G8B8A8_SRGB,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        view_type: vk::ImageViewType::TYPE_2D,
    };
}
