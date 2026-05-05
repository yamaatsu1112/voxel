pub(crate) use crate::builders::configs::pipeline_configs::dynamic_voxel::{
    DYNAMIC_VOXEL_DESCRIPTOR_SET_LAYOUT_BINDINGS, DYNAMIC_VOXEL_PUSH_CONSTANT_RANGES,
};
pub(crate) use crate::builders::configs::pipeline_configs::voxel::{
    COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS, VOXEL_PUSH_CONSTANT_RANGES,
};
pub(crate) use crate::builders::configs::pipeline_configs::{
    CollisionComputePipeline, RaycastComputePipeline,
};
use crate::vulkan::pipeline_config::ComputePipelineConfig;

macro_rules! individual_shader_path {
    ($name:literal) => {
        concat!(env!("VOXEL_SHADER_DIR"), "/", $name)
    };
}

macro_rules! define_individual_edit_pipeline {
    ($marker:ident, $shader:literal) => {
        engine_macro::define_compute_pipeline! {
            pub struct $marker;
            config = ComputePipelineConfig {
                shader_path: individual_shader_path!($shader),
                descriptor_set_layouts: &[&COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS],
                push_constant_ranges: &VOXEL_PUSH_CONSTANT_RANGES,
            };
        }
    };
}

define_individual_edit_pipeline!(
    VoxelCollectMaterializePipeline,
    "voxel_collect_materialize_individual.spv"
);
define_individual_edit_pipeline!(
    VoxelScanNodeOffsetsPipeline,
    "voxel_scan_node_offsets_individual.spv"
);
define_individual_edit_pipeline!(
    VoxelScanLeafOffsetsPipeline,
    "voxel_scan_leaf_offsets_individual.spv"
);
define_individual_edit_pipeline!(
    VoxelSnapshotAllocationPipeline,
    "voxel_snapshot_allocation_individual.spv"
);
define_individual_edit_pipeline!(
    VoxelInitializeMaterializedPipeline,
    "voxel_initialize_materialized_individual.spv"
);
define_individual_edit_pipeline!(
    VoxelLinkMaterializedPipeline,
    "voxel_link_materialized_individual.spv"
);
define_individual_edit_pipeline!(
    VoxelCommitMaterializedPipeline,
    "voxel_commit_materialized_individual.spv"
);
define_individual_edit_pipeline!(VoxelApplyPlacePipeline, "voxel_apply_place_individual.spv");
define_individual_edit_pipeline!(
    VoxelApplyDestroyPipeline,
    "voxel_apply_destroy_individual.spv"
);
define_individual_edit_pipeline!(
    VoxelCollectFreePipeline,
    "voxel_collect_free_individual.spv"
);
define_individual_edit_pipeline!(
    VoxelScanFreeOffsetsPipeline,
    "voxel_scan_free_offsets_individual.spv"
);
define_individual_edit_pipeline!(VoxelFreeBatchPipeline, "voxel_free_batch_individual.spv");

engine_macro::define_compute_pipeline! {
    pub struct DynamicVoxelClearPipeline;
    config = ComputePipelineConfig {
        shader_path: individual_shader_path!("dynamic_voxel_clear_individual.spv"),
        descriptor_set_layouts: &[&DYNAMIC_VOXEL_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &DYNAMIC_VOXEL_PUSH_CONSTANT_RANGES,
    };
}
