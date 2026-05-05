use ash::vk;

use crate::vulkan::resource_config::SamplerConfig;

engine_macro::define_sampler! {
    pub struct UITextureSampler;
    config = SamplerConfig {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        address_mode: vk::SamplerAddressMode::REPEAT,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
    };
}
