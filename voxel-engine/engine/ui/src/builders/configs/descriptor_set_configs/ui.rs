use crate::builders::configs::pipeline_configs::UI_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// UI descriptor set for HUD buffers.
    pub struct UIDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &UI_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
