use crate::builders::configs::pipeline_configs::RAYCAST_COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// Descriptor set for the raycast compute pass.
    pub struct RaycastComputeDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &RAYCAST_COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
