use crate::builders::configs::pipeline_configs::MAIN_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// Main descriptor set bindings used every frame.
    pub struct MainDescriptorSet;
    lifetime = PerFrame;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &MAIN_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
