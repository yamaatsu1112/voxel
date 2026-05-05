use ash::vk;

use crate::vulkan::resource_config::SamplerConfig;

engine_macro::define_sampler! {
    pub struct VoxelTexture3DSampler;
    config = SamplerConfig {
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
    };
}
