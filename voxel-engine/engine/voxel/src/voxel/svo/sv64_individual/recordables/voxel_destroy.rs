use ash::vk;

use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;
use crate::voxel::svo::MAX_DEPTH;
use crate::voxel::svo::sv64_individual::buffers::DestroyIndirectDispatchBuffer;
use crate::voxel::svo::sv64_individual::descriptor_sets::{
    PrepareDestroyDispatchDescriptorSet, VoxelDestroyDescriptorSet,
};
use crate::voxel::svo::sv64_individual::pipelines::{
    LAYER_TYPE_ALLOCATE_BATCH, LAYER_TYPE_COLLECT_FREE_REQUESTS, LAYER_TYPE_COLLECT_REQUESTS,
    LAYER_TYPE_FREE_BATCH, VoxelDestroyPipeline,
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
pub struct VoxelDestroyPipelineRecordable;

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyPrepareDispatchCB;
    count = 1;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyCollectRequestsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyComputeUniqueFlagsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyPrefixSumCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyCompactUniqueParirsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyAllocateBatchCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyUpdateLeavesCB;
    count = 1;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyCollectFreeRequestsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct VoxelDestroyFreeBatchCB;
    count = MAX_DEPTH as usize;
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
            VoxelDestroyCollectRequestsCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_collect_requests_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
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
            VoxelDestroyComputeUniqueFlagsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compute_unique_flags_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelDestroyPrefixSumCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_prefix_sum_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelDestroyCompactUniqueParirsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compact_unique_pairs_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelDestroyAllocateBatchCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_batch_nodes_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(
                        secondary_context, depth, svo_idx, LAYER_TYPE_ALLOCATE_BATCH
                    );
                }
            }
        );
    }

    let secondary_buffer = get_reusable_secondary_command_buffer::<VoxelDestroyUpdateLeavesCB, _>(
        context,
        vk::CommandBufferUsageFlags::empty(),
        0,
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
            for svo_idx in 0..NUM_SVO_DATA {
                record_update_leaves_for_svo::<
                    VoxelDestroyPipeline,
                    VoxelDestroyDescriptorSet,
                    DestroyIndirectDispatchBuffer,
                >(secondary_context, svo_idx);
            }
        },
    );
    secondary_buffers.push(secondary_buffer);

    for depth in (1..=MAX_DEPTH).rev() {
        let depth_index = depth as usize - 1;

        push_depth_buffer!(
            VoxelDestroyCollectFreeRequestsCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_collect_requests_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
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
            VoxelDestroyComputeUniqueFlagsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compute_unique_flags_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelDestroyPrefixSumCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_prefix_sum_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelDestroyCompactUniqueParirsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compact_unique_pairs_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            VoxelDestroyFreeBatchCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<VoxelDestroyPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_batch_nodes_for_svo::<
                        VoxelDestroyPipeline,
                        VoxelDestroyDescriptorSet,
                        DestroyIndirectDispatchBuffer,
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
        record_compute(context);
    }
}
