use ash::vk;

use crate::vulkan::resource_config::{ImageConfig, ImageSizeFormat, ImageViewConfig};

engine_macro::define_image! {
    /// 3D voxel texture image (8x8x2048, RGBA8)
    pub struct VoxelTexture3DImage;
    lifetime = Persistent;
    count = 1;
    config = ImageConfig {
        size: ImageSizeFormat::Fixed3D {
            width: 8,
            height: 8,
            depth: 2048,
        },
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::from_raw(
            vk::ImageUsageFlags::TRANSFER_DST.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_image_view! {
    /// 3D voxel texture image view
    pub struct VoxelTexture3DImageView;
    image = VoxelTexture3DImage;
    lifetime = Persistent;
    count = 1;
    config = ImageViewConfig {
        format: vk::Format::R8G8B8A8_UNORM,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        view_type: vk::ImageViewType::TYPE_3D,
    };
}
