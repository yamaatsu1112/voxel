use crate::builders::configs::pipeline_configs::prepare_dispatch::PREPARE_DISPATCH_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vulkan::resource_config::DescriptorSetConfig;
use engine_common::EngineDescriptorPool;

engine_macro::define_descriptor_set! {
    pub struct PrepareVoxelDispatchDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &PREPARE_DISPATCH_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}

engine_macro::define_descriptor_set! {
    pub struct PrepareDestroyDispatchDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &PREPARE_DISPATCH_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}

engine_macro::define_descriptor_set! {
    pub struct PrepareDynamicDispatchDescriptorSet;
    lifetime = Persistent;
    pool = EngineDescriptorPool;
    config = DescriptorSetConfig {
        layout_bindings: &PREPARE_DISPATCH_DESCRIPTOR_SET_LAYOUT_BINDINGS,
    };
}
