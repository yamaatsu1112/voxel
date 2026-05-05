use crate::builders::configs::pipeline_configs::COLLISION_COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// Descriptor set for collision compute inputs.
    pub struct CollisionComputeDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &COLLISION_COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
