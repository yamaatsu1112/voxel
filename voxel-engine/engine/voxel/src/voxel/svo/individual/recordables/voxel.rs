use ash::vk;

use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;
use crate::voxel::svo::MAX_DEPTH;
use crate::voxel::svo::individual::buffers::IndirectDispatchBuffer;
use crate::voxel::svo::individual::descriptor_sets::{
    PrepareVoxelDispatchDescriptorSet, VoxelDescriptorSet,
};
use crate::voxel::svo::individual::pipelines::{
    VoxelApplyPlacePipeline, VoxelCollectFreePipeline, VoxelCollectMaterializePipeline,
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
pub struct VoxelPipelineRecordable;

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelPrepareDispatchCB;
    count = 1;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDepthStageCB;
    count = (MAX_DEPTH as usize) * 9;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelApplyPlaceCB;
    count = 1;
}

fn depth_stage_index(depth: u32, stage: usize) -> usize {
    ((depth - 1) as usize) * 9 + stage
}

fn execute_depth_stage<M, F>(
    context: &mut VkComputeRecordContext,
    index: usize,
    usage_flags: vk::CommandBufferUsageFlags,
    record_fn: F,
) where
    M: crate::vulkan::resource_config::ReusableCommandBufferMarker,
    F: FnOnce(&crate::vulkan::record_context::VkSecondaryRecordContext),
{
    let secondary_buffer =
        get_reusable_secondary_command_buffer::<M, _>(context, usage_flags, index, record_fn);
    context.cmd_execute_commands(&[secondary_buffer]);
    insert_memory_barrier_to_buffer(context);
}

fn record_compact_all_depth_allocation(context: &mut VkComputeRecordContext) {
    let mut stage = 0usize;

    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(1, stage),
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelCollectMaterializePipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelCollectMaterializePipeline,
                    VoxelDescriptorSet,
                    IndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_PADDED);
            }
        },
    );
    stage += 1;

    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(1, stage),
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelScanNodeOffsetsPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                for (phase, slot) in [(0, SLOT_PADDED), (1, SLOT_ONE), (2, SLOT_PADDED)] {
                    record_edit_stage_for_svo::<
                        VoxelScanNodeOffsetsPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, 0, phase, svo_index, slot);
                    insert_memory_barrier_to_secondary_buffer(secondary_context);
                }
            }
        },
    );
    stage += 1;

    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(1, stage),
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelScanLeafOffsetsPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                for (phase, slot) in [(0, SLOT_PADDED), (1, SLOT_ONE), (2, SLOT_PADDED)] {
                    record_edit_stage_for_svo::<
                        VoxelScanLeafOffsetsPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, 0, phase, svo_index, slot);
                    insert_memory_barrier_to_secondary_buffer(secondary_context);
                }
            }
        },
    );
    stage += 1;

    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(1, stage),
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelSnapshotAllocationPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelSnapshotAllocationPipeline,
                    VoxelDescriptorSet,
                    IndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_ONE);
            }
        },
    );
    stage += 1;

    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(1, stage),
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelInitializeMaterializedPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelInitializeMaterializedPipeline,
                    VoxelDescriptorSet,
                    IndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_UNPADDED);
            }
        },
    );
    stage += 1;

    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(1, stage),
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelLinkMaterializedPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelLinkMaterializedPipeline,
                    VoxelDescriptorSet,
                    IndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_UNPADDED);
            }
        },
    );
    stage += 1;

    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(1, stage),
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelCommitMaterializedPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_one_thread_stage_for_svo::<
                    VoxelCommitMaterializedPipeline,
                    VoxelDescriptorSet,
                >(secondary_context, 0, 0, svo_index);
            }
        },
    );
}

fn record_apply_place(context: &mut VkComputeRecordContext) {
    execute_depth_stage::<VoxelApplyPlaceCB, _>(
        context,
        0,
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelApplyPlacePipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelApplyPlacePipeline,
                    VoxelDescriptorSet,
                    IndirectDispatchBuffer,
                >(secondary_context, 0, 0, svo_index, SLOT_UNPADDED);
            }
        },
    );
}

fn record_collapse_depth(context: &mut VkComputeRecordContext, depth: u32) {
    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(depth, 7),
        vk::CommandBufferUsageFlags::empty(),
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelCollectFreePipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelCollectFreePipeline,
                    VoxelDescriptorSet,
                    IndirectDispatchBuffer,
                >(secondary_context, depth, 0, svo_index, SLOT_PADDED);
            }
        },
    );

    execute_depth_stage::<VoxelDepthStageCB, _>(
        context,
        depth_stage_index(depth, 8),
        vk::CommandBufferUsageFlags::empty(),
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
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, depth, phase, svo_index, slot);
                    insert_memory_barrier_to_secondary_buffer(secondary_context);
                }
            }
            secondary_context.cmd_bind_compute_pipeline::<VoxelFreeBatchPipeline>();
            for svo_index in 0..NUM_SVO_DATA {
                record_edit_stage_for_svo::<
                    VoxelFreeBatchPipeline,
                    VoxelDescriptorSet,
                    IndirectDispatchBuffer,
                >(secondary_context, depth, 0, svo_index, SLOT_PADDED);
            }
        },
    );
}

impl VoxelPipelineRecordable {
    pub fn new() -> Self {
        Self
    }
}

impl VkComputeRecordable for VoxelPipelineRecordable {
    fn record(context: &mut VkComputeRecordContext) {
        context.cmd_begin_timing("VoxelPipelineRecordable");
        record_prepare_dispatch::<PrepareVoxelDispatchDescriptorSet, VoxelPrepareDispatchCB>(
            context,
        );

        record_compact_all_depth_allocation(context);
        record_apply_place(context);
        for depth in (1..=MAX_DEPTH).rev() {
            record_collapse_depth(context, depth);
        }
        context.cmd_end_timing("VoxelPipelineRecordable");
    }
}
