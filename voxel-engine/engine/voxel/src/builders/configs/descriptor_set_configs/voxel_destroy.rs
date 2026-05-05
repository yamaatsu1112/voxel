use crate::builders::configs::pipeline_configs::COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    /// Descriptor set that covers voxel destroy compute buffers.
    pub struct VoxelDestroyDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
