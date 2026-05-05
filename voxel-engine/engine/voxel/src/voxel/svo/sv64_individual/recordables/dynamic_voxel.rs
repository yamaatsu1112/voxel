use ash::vk;

use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;
use crate::voxel::svo::MAX_DEPTH;
use crate::voxel::svo::sv64_individual::buffers::DynamicIndirectDispatchBuffer;
use crate::voxel::svo::sv64_individual::descriptor_sets::{
    DynamicVoxelDescriptorSet, PrepareDynamicDispatchDescriptorSet,
};
use crate::voxel::svo::sv64_individual::pipelines::{
    DynamicVoxelPipeline, LAYER_TYPE_ALLOCATE_BATCH, LAYER_TYPE_CLEAR_SVO,
    LAYER_TYPE_COLLECT_FREE_REQUESTS, LAYER_TYPE_COLLECT_REQUESTS, LAYER_TYPE_FREE_BATCH,
    VoxelPushConstants,
};
use crate::vulkan::record_context::{ShaderStageFlags, VkComputeRecordContext};
use crate::vulkan::recordable::VkComputeRecordable;

use super::common::{
    get_reusable_secondary_command_buffer, insert_memory_barrier_to_buffer,
    record_batch_nodes_for_svo, record_collect_requests_for_svo,
    record_compact_unique_pairs_for_svo, record_compute_unique_flags_for_svo,
    record_prefix_sum_for_svo, record_prepare_dispatch, record_update_leaves_for_svo,
};

#[derive(Clone, Default)]
pub struct DynamicVoxelPipelineRecordable;

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelClearCB;
    count = 1;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelPrepareDispatchCB;
    count = 1;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelCollectRequestsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelComputeUniqueFlagsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelPrefixSumCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelCompactUniqueParirsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelAllocateBatchCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelUpdateLeavesCB;
    count = 1;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelCollectFreeRequestsCB;
    count = MAX_DEPTH as usize;
}

engine_macro::define_reusable_command_buffer! {
    pub struct DynamicVoxelFreeBatchCB;
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
            DynamicVoxelCollectRequestsCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_collect_requests_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
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
            DynamicVoxelComputeUniqueFlagsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compute_unique_flags_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            DynamicVoxelPrefixSumCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_prefix_sum_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            DynamicVoxelCompactUniqueParirsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compact_unique_pairs_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            DynamicVoxelAllocateBatchCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_batch_nodes_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
                    >(
                        secondary_context, depth, svo_idx, LAYER_TYPE_ALLOCATE_BATCH
                    );
                }
            }
        );
    }

    let secondary_buffer = get_reusable_secondary_command_buffer::<DynamicVoxelUpdateLeavesCB, _>(
        context,
        vk::CommandBufferUsageFlags::empty(),
        0,
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
            for svo_idx in 0..NUM_SVO_DATA {
                record_update_leaves_for_svo::<
                    DynamicVoxelPipeline,
                    DynamicVoxelDescriptorSet,
                    DynamicIndirectDispatchBuffer,
                >(secondary_context, svo_idx);
            }
        },
    );
    secondary_buffers.push(secondary_buffer);

    for depth in (1..=MAX_DEPTH).rev() {
        let depth_index = depth as usize - 1;

        push_depth_buffer!(
            DynamicVoxelCollectFreeRequestsCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_collect_requests_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
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
            DynamicVoxelComputeUniqueFlagsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compute_unique_flags_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            DynamicVoxelPrefixSumCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_prefix_sum_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            DynamicVoxelCompactUniqueParirsCB,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_compact_unique_pairs_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
                    >(secondary_context, depth, svo_idx);
                }
            }
        );

        push_depth_buffer!(
            DynamicVoxelFreeBatchCB,
            vk::CommandBufferUsageFlags::empty(),
            depth_index,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_idx in 0..NUM_SVO_DATA {
                    record_batch_nodes_for_svo::<
                        DynamicVoxelPipeline,
                        DynamicVoxelDescriptorSet,
                        DynamicIndirectDispatchBuffer,
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

impl DynamicVoxelPipelineRecordable {
    pub fn new() -> Self {
        Self
    }

    fn record_clear(context: &mut VkComputeRecordContext) {
        let secondary_buffer = get_reusable_secondary_command_buffer::<DynamicVoxelClearCB, _>(
            context,
            vk::CommandBufferUsageFlags::empty(),
            0,
            |secondary_context| {
                secondary_context.cmd_bind_compute_pipeline::<DynamicVoxelPipeline>();
                for svo_index in 0..NUM_SVO_DATA {
                    let push_constants = VoxelPushConstants {
                        depth: 0,
                        layer_type: LAYER_TYPE_CLEAR_SVO,
                        phase: 0,
                        svo_index: svo_index as u32,
                    };
                    secondary_context
                        .cmd_bind_descriptor_set::<DynamicVoxelPipeline, DynamicVoxelDescriptorSet>(
                            0,
                            &[],
                        );
                    secondary_context.cmd_push_constants::<DynamicVoxelPipeline, _>(
                        ShaderStageFlags::COMPUTE,
                        0,
                        &push_constants,
                    );
                    secondary_context.cmd_dispatch(1, 1, 1);
                }
            },
        );

        context.cmd_execute_commands(&[secondary_buffer]);
        insert_memory_barrier_to_buffer(context);
    }
}

impl VkComputeRecordable for DynamicVoxelPipelineRecordable {
    fn record(context: &mut VkComputeRecordContext) {
        Self::record_clear(context);
        record_prepare_dispatch::<PrepareDynamicDispatchDescriptorSet, DynamicVoxelPrepareDispatchCB>(
            context,
        );
        record_compute(context);
    }
}
