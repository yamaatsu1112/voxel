use crate::builders::configs::pipeline_configs::MAIN_STATIC_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// Persistent descriptor set for static renderer bindings.
    pub struct MainStaticDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &MAIN_STATIC_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
