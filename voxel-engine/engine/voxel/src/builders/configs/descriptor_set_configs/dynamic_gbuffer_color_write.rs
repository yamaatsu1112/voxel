use crate::builders::configs::pipeline_configs::DYNAMIC_GBUFFER_COLOR_WRITE_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// Per-frame descriptor set for dynamic GBuffer color writes.
    pub struct DynamicGBufferColorWriteDescriptorSet;
    lifetime = PerFrame;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &DYNAMIC_GBUFFER_COLOR_WRITE_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
