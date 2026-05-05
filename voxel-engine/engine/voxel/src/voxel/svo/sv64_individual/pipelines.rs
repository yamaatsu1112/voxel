pub(crate) use crate::builders::configs::pipeline_configs::dynamic_voxel::{
    DYNAMIC_VOXEL_DESCRIPTOR_SET_LAYOUT_BINDINGS, DYNAMIC_VOXEL_PUSH_CONSTANT_RANGES,
    LAYER_TYPE_CLEAR_SVO,
};
pub(crate) use crate::builders::configs::pipeline_configs::voxel::{
    COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS, LAYER_TYPE_ALLOCATE_BATCH,
    LAYER_TYPE_COLLECT_FREE_REQUESTS, LAYER_TYPE_COLLECT_REQUESTS, LAYER_TYPE_COMPACT,
    LAYER_TYPE_FREE_BATCH, LAYER_TYPE_PREFIX_SUM, LAYER_TYPE_UNIQUE_FLAGS,
    LAYER_TYPE_UPDATE_LEAVES, VOXEL_PUSH_CONSTANT_RANGES, VoxelPushConstants,
};
pub(crate) use crate::builders::configs::pipeline_configs::{
    CollisionComputePipeline, RaycastComputePipeline,
};
use crate::vulkan::pipeline_config::ComputePipelineConfig;

const SV64_INDIVIDUAL_VOXEL_SHADER_PATH: &str =
    concat!(env!("VOXEL_SHADER_DIR"), "/voxel_sv64_individual.spv");
const SV64_INDIVIDUAL_DYNAMIC_VOXEL_SHADER_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/dynamic_voxel_sv64_individual.spv"
);
const SV64_INDIVIDUAL_VOXEL_DESTROY_SHADER_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/voxel_destroy_sv64_individual.spv"
);

engine_macro::define_compute_pipeline! {
    pub struct VoxelPipeline;
    config = ComputePipelineConfig {
        shader_path: SV64_INDIVIDUAL_VOXEL_SHADER_PATH,
        descriptor_set_layouts: &[&COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &VOXEL_PUSH_CONSTANT_RANGES,
    };
}

engine_macro::define_compute_pipeline! {
    pub struct DynamicVoxelPipeline;
    config = ComputePipelineConfig {
        shader_path: SV64_INDIVIDUAL_DYNAMIC_VOXEL_SHADER_PATH,
        descriptor_set_layouts: &[&DYNAMIC_VOXEL_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &DYNAMIC_VOXEL_PUSH_CONSTANT_RANGES,
    };
}

engine_macro::define_compute_pipeline! {
    pub struct VoxelDestroyPipeline;
    config = ComputePipelineConfig {
        shader_path: SV64_INDIVIDUAL_VOXEL_DESTROY_SHADER_PATH,
        descriptor_set_layouts: &[&COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &VOXEL_PUSH_CONSTANT_RANGES,
    };
}
