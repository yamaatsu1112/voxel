use crate::builders::configs::pipeline_configs::DYNAMIC_VOXEL_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// Descriptor set for dynamic voxel compute passes.
    pub struct DynamicVoxelDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &DYNAMIC_VOXEL_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
