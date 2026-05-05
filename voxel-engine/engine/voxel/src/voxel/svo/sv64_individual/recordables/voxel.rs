use ash::vk;

use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;
use crate::voxel::svo::MAX_DEPTH;
use crate::voxel::svo::sv64_individual::buffers::IndirectDispatchBuffer;
use crate::voxel::svo::sv64_individual::descriptor_sets::{
    PrepareVoxelDispatchDescriptorSet, VoxelDescriptorSet,
};
use crate::voxel::svo::sv64_individual::pipelines::{
    LAYER_TYPE_ALLOCATE_BATCH, LAYER_TYPE_COLLECT_FREE_REQUESTS, LAYER_TYPE_COLLECT_REQUESTS,
    LAYER_TYPE_FREE_BATCH, VoxelPipeline,
};
use crate::vulkan::record_context::VkComputeRecordContext;
use crate::vulkan::recordable::VkComputeRecordable;

use super::common::{
    get_reusable_secondary_command_buffer, insert_memory_barrier_to_buffer,
    record_batch_nodes_for_svo, record_collect_requests_for_svo,
    record_compact_unique_pairs_for_svo, record_compute_unique_flags_for_svo,
    record_prefix_sum_for_svo, record_prepare_dispatch, record_update_leaves_for_svo,
};

#[derive(Clone, Default)]
pub struct VoxelPipelineRecordable;

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelCollectRequestsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelComputeUniqueFlagsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelPrefixSumCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelCompactUniqueParirsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelAllocateBatchCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelUpdateLeavesCB;
    count = 1;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelCollectFreeRequestsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelFreeBatchCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelPrepareDispatchCB;
    count = 1;
}

fn record_compute(context: &mut VkComputeRecordContext) {
    let mut secondary_buffers = Vec::new();

    macro_rules! push_depth_buffer {
        ($marker:ty, $usage_flags:expr, $depth_index:expr, |$secondary_context:ident| $body:block) => {{
            let secondary_buffer = get_reusable_secondary_command_buffer::<$marker, _>(
                context,
                $usage_flags,
                $depth_index,
                |$secondary_context| $body,
            );
            secondary_buffers.push(secondary_buffer);
        }};
    }

    for depth in 1..=MAX_DEPTH {
        let depth_index = depth as usize - 1;

        push_depth_buffer!(
            VoxelCollectRequestsCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_collect_requests_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(
                        secondary_context,
                        depth,
                        svo_idx,
                        LAYER_TYPE_COLLECT_REQUESTS,
                    );
                }
            }
        );

        push_depth_buffer!(
            VoxelComputeUniqueFlagsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compute_unique_flags_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelPrefixSumCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_prefix_sum_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelCompactUniqueParirsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compact_unique_pairs_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelAllocateBatchCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_batch_nodes_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(
                        secondary_context, depth, svo_idx, LAYER_TYPE_ALLOCATE_BATCH
                    );
                }
            }
        );
    }

    let secondary_buffer = get_reusable_secondary_command_buffer::<VoxelUpdateLeavesCB, _>(
        context,
        vk::CommandBufferUsageFlags::empty(),
        0,
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
            for svo_idx in 0..NUM_SVO_DATA {
                record_update_leaves_for_svo::<
                    VoxelPipeline,
                    VoxelDescriptorSet,
                    IndirectDispatchBuffer,
                >(secondary_context, svo_idx);
            }
        },
    );
    secondary_buffers.push(secondary_buffer);

    for depth in (1..=MAX_DEPTH).rev() {
        let depth_index = depth as usize - 1;

        push_depth_buffer!(
            VoxelCollectFreeRequestsCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_collect_requests_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(
                        secondary_context,
                        depth,
                        svo_idx,
                        LAYER_TYPE_COLLECT_FREE_REQUESTS,
                    );
                }
            }
        );

        push_depth_buffer!(
            VoxelComputeUniqueFlagsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compute_unique_flags_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelPrefixSumCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_prefix_sum_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelCompactUniqueParirsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compact_unique_pairs_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelFreeBatchCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_batch_nodes_for_svo::<
                        VoxelPipeline,
                        VoxelDescriptorSet,
                        IndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx, LAYER_TYPE_FREE_BATCH);
                }
            }
        );
    }

    for secondary_buffer in &secondary_buffers {
        context.cmd_execute_commands(std::slice::from_ref(secondary_buffer));
        insert_memory_barrier_to_buffer(context);
    }
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
        record_compute(context);
        context.cmd_end_timing("VoxelPipelineRecordable");
    }
}
