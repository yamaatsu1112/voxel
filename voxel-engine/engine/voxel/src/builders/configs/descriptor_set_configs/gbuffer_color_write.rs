use crate::builders::configs::pipeline_configs::GBUFFER_COLOR_WRITE_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// Per-frame descriptor set for GBuffer color write.
    pub struct GBufferColorWriteDescriptorSet;
    lifetime = PerFrame;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &GBUFFER_COLOR_WRITE_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
