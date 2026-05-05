use ash::vk;

use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;
use crate::voxel::svo::MAX_DEPTH;
use crate::voxel::svo::individual::buffers::DestroyIndirectDispatchBuffer;
use crate::voxel::svo::individual::descriptor_sets::{
    PrepareDestroyDispatchDescriptorSet, VoxelDestroyDescriptorSet,
};
use crate::voxel::svo::individual::pipelines::{
    VoxelApplyDestroyPipeline, VoxelCollectFreePipeline, VoxelCollectMaterializePipeline,
    VoxelCommitMaterializedPipeline, VoxelFreeBatchPipeline, VoxelInitializeMaterializedPipeline,
    VoxelLinkMaterializedPipeline, VoxelScanFreeOffsetsPipeline, VoxelScanLeafOffsetsPipeline,
    VoxelScanNodeOffsetsPipeline, VoxelSnapshotAllocationPipeline,
};
use crate::vulkan::record_context::VkComputeRecordContext;
use crate::vulkan::recordable::VkComputeRecordable;

use super::common::{
    SLOT_ONE, SLOT_PADDED, SLOT_UNPADDED, get_reusable_secondary_command_buffer,
    insert_memory_barrier_to_buffer, insert_memory_barrier_to_secondary_buffer,
    record_edit_stage_for_svo, record_one_thread_stage_for_svo, record_prepare_dispatch,
};

#[derive(Clone, Default)]
pub struct VoxelDestroyPipelineRecordable;

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyPrepareDispatchCB;
    count = 1;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyDepthStageCB;
    count = (MAX_DEPTH as usize) * 9;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyApplyCB;
    count = 1;
}

fn depth_stage_index(depth: u32, stage: usize) -> usize {
    ((depth - 1) as usize) * 9 + stage
}

fn execute_stage<M, F>(context: &mut VkComputeRecordContext, index: usize, record_fn: F)
where
    M: crate::vulkan::resource_config::ReusableCommandBufferMarker,
    F: FnOnce(&crate::vulkan::record_context::VkSecondaryRecordContext),
{
    let secondary_buffer = get_reusable_secondary_command_buffer::<M, _>(
        context,
        vk::CommandBufferUsageFlags::empty(),
        index,
        record_fn,
    );
    context.cmd_execute_commands(&[secondary_buffer]);
    insert_memory_barrier_to_buffer(context);
}

fn record_compact_all_depth_allocation(context: &mut VkComputeRecordContext) {
    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(1, 0),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelCollectMaterializePipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelCollectMaterializePipeline,
                    VoxelDestroyDescriptorSet,
                    DestroyIndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_PADDED);
            }
        },
    );

    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(1, 1),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelScanNodeOffsetsPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                for (phase, slot) in [(0, SLOT_PADDED), (1, SLOT_ONE), (2, SLOT_PADDED)] {
                    record_edit_stage_for_svo::<
                        VoxelScanNodeOffsetsPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, 0, phase, svo_index, slot);
                    insert_memory_barrier_to_secondary_buffer(secondary_context);
                }
            }
        },
    );

    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(1, 2),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelScanLeafOffsetsPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                for (phase, slot) in [(0, SLOT_PADDED), (1, SLOT_ONE), (2, SLOT_PADDED)] {
                    record_edit_stage_for_svo::<
                        VoxelScanLeafOffsetsPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, 0, phase, svo_index, slot);
                    insert_memory_barrier_to_secondary_buffer(secondary_context);
                }
            }
        },
    );

    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(1, 3),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelSnapshotAllocationPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelSnapshotAllocationPipeline,
                    VoxelDestroyDescriptorSet,
                    DestroyIndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_ONE);
            }
        },
    );

    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(1, 4),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelInitializeMaterializedPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelInitializeMaterializedPipeline,
                    VoxelDestroyDescriptorSet,
                    DestroyIndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_UNPADDED);
            }
        },
    );

    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(1, 5),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelLinkMaterializedPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelLinkMaterializedPipeline,
                    VoxelDestroyDescriptorSet,
                    DestroyIndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_UNPADDED);
            }
        },
    );

    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(1, 6),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelCommitMaterializedPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_one_thread_stage_for_svo::<
                    VoxelCommitMaterializedPipeline,
                    VoxelDestroyDescriptorSet,
                >(secondary_context, 0, 0, svo_index);
            }
        },
    );
}

fn record_collapse_depth(context: &mut VkComputeRecordContext, depth: u32) {
    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(depth, 7),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelCollectFreePipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelCollectFreePipeline,
                    VoxelDestroyDescriptorSet,
                    DestroyIndirectDispatchBuffer,
                >(secondary_context, depth, 0, svo_index, SLOT_PADDED);
            }
        },
    );

    execute_stage::<VoxelDestroyDepthStageCB, _>(
        context,
        depth_stage_index(depth, 8),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelScanFreeOffsetsPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                for (phase, slot) in [
                    (0, SLOT_PADDED),
                    (1, SLOT_PADDED),
                    (2, SLOT_ONE),
                    (3, SLOT_PADDED),
                    (4, SLOT_PADDED),
                ] {
                    record_edit_stage_for_svo::<
                        VoxelScanFreeOffsetsPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, depth, phase, svo_index, slot);
                    insert_memory_barrier_to_secondary_buffer(secondary_context);
                }
            }
            secondary_context.cmd_bind_compute_pipeline::<VoxelFreeBatchPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelFreeBatchPipeline,
                    VoxelDestroyDescriptorSet,
                    DestroyIndirectDispatchBuffer,
                >(secondary_context, depth, 0, svo_index, SLOT_PADDED);
            }
        },
    );
}

impl VoxelDestroyPipelineRecordable {
    pub fn new() -> Self {
        Self
    }
}

impl VkComputeRecordable for VoxelDestroyPipelineRecordable {
    fn record(context: &mut VkComputeRecordContext) {
        record_prepare_dispatch::<PrepareDestroyDispatchDescriptorSet, VoxelDestroyPrepareDispatchCB>(
            context,
        );
        record_compact_all_depth_allocation(context);
        execute_stage::<VoxelDestroyApplyCB, _>(context, 0, |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelApplyDestroyPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelApplyDestroyPipeline,
                    VoxelDestroyDescriptorSet,
                    DestroyIndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_UNPADDED);
            }
        });
        for depth in (1..=MAX_DEPTH).rev() {
            record_collapse_depth(context, depth);
        }
    }
}
