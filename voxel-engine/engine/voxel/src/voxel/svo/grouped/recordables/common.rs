use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::builders::configs::buffer_configs::compute::INDIRECT_DISPATCH_SLOTS_PER_SVO;
use crate::voxel::svo::grouped::pipelines::{
    LAYER_TYPE_COMPACT, LAYER_TYPE_PREFIX_SUM, LAYER_TYPE_UNIQUE_FLAGS, LAYER_TYPE_UPDATE_LEAVES,
    VoxelPushConstants,
};
use crate::vulkan::record_context::{
    DescriptorSetResolver, SecondaryCommandBufferHandle, ShaderStageFlags, VkComputeRecordContext,
    VkSecondaryRecordContext,
};
use crate::vulkan::resource_config::{
    BufferMarker, ComputePipelineMarker, DescriptorSetMarker, ReusableCommandBufferMarker,
};
use crate::vulkan::resource_lifetime::Persistent;

const SLOT_PADDED: usize = 0;
const SLOT_ONE: usize = 1;
const SLOT_UNPADDED: usize = 2;
const DISPATCH_INDIRECT_STRIDE: u64 = std::mem::size_of::<vk::DispatchIndirectCommand>() as u64;

pub(crate) fn get_reusable_secondary_command_buffer<M, F>(
    context: &mut VkComputeRecordContext,
    usage_flags: vk::CommandBufferUsageFlags,
    index: usize,
    record_fn: F,
) -> SecondaryCommandBufferHandle
where
    M: ReusableCommandBufferMarker,
    F: FnOnce(&VkSecondaryRecordContext),
{
    let (secondary_context, is_new) = context.get_reusable_secondary::<M>(index);
    if is_new {
        secondary_context.begin_with_flags(usage_flags);
        record_fn(&secondary_context);
        secondary_context.end();
    }
    secondary_context.handle()
}

pub(crate) fn insert_memory_barrier_to_buffer(context: &VkComputeRecordContext) {
    let memory_barrier = vk::MemoryBarrier {
        s_type: vk::StructureType::MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::SHADER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        _marker: PhantomData,
    };
    context.cmd_pipeline_barrier(
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[memory_barrier],
        &[],
        &[],
    );
}

fn insert_memory_barrier_to_secondary_buffer(context: &VkSecondaryRecordContext) {
    let memory_barrier = vk::MemoryBarrier {
        s_type: vk::StructureType::MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::SHADER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        _marker: PhantomData,
    };
    context.cmd_pipeline_barrier(
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[memory_barrier],
        &[],
        &[],
    );
}

pub(crate) fn insert_memory_barrier_after_prepare(context: &VkComputeRecordContext) {
    let memory_barrier = vk::MemoryBarrier {
        s_type: vk::StructureType::MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::SHADER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::INDIRECT_COMMAND_READ,
        _marker: PhantomData,
    };
    context.cmd_pipeline_barrier(
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        &[memory_barrier],
        &[],
        &[],
    );
}

pub(crate) fn record_prepare_dispatch<PD, M>(context: &mut VkComputeRecordContext)
where
    PD: DescriptorSetMarker,
    M: ReusableCommandBufferMarker,
    PD::Lifetime: DescriptorSetResolver,
{
    use crate::builders::configs::pipeline_configs::PrepareDispatchPipeline;

    let secondary_buffer = get_reusable_secondary_command_buffer::<M, _>(
        context,
        vk::CommandBufferUsageFlags::empty(),
        0,
        |secondary_context| {
            secondary_context.cmd_bind_compute_pipeline::<PrepareDispatchPipeline>();
            secondary_context.cmd_bind_descriptor_set::<PrepareDispatchPipeline, PD>(0, &[]);
            secondary_context.cmd_dispatch(1, 1, 1);
        },
    );

    context.cmd_execute_commands(&[secondary_buffer]);
    insert_memory_barrier_after_prepare(context);
}

pub(crate) fn record_collect_requests_for_svo<P, D, I>(
    context: &VkSecondaryRecordContext,
    depth: u32,
    svo_index: usize,
    layer_type: u32,
) where
    P: ComputePipelineMarker,
    D: DescriptorSetMarker,
    I: BufferMarker<Lifetime = Persistent>,
    D::Lifetime: DescriptorSetResolver,
{
    context.cmd_bind_descriptor_set::<P, D>(0, &[]);

    let push_constants = VoxelPushConstants {
        depth,
        layer_type,
        phase: 0,
        svo_index: svo_index as u32,
    };
    context.cmd_push_constants::<P, _>(ShaderStageFlags::COMPUTE, 0, &push_constants);
    dispatch_indirect_for_svo::<I>(context, svo_index, SLOT_PADDED);
}

pub(crate) fn record_compute_unique_flags_for_svo<P, D, I>(
    context: &VkSecondaryRecordContext,
    depth: u32,
    svo_index: usize,
) where
    P: ComputePipelineMarker,
    D: DescriptorSetMarker,
    I: BufferMarker<Lifetime = Persistent>,
    D::Lifetime: DescriptorSetResolver,
{
    context.cmd_bind_descriptor_set::<P, D>(0, &[]);

    let push_constants = VoxelPushConstants {
        depth,
        layer_type: LAYER_TYPE_UNIQUE_FLAGS,
        phase: 0,
        svo_index: svo_index as u32,
    };

    context.cmd_push_constants::<P, _>(ShaderStageFlags::COMPUTE, 0, &push_constants);
    dispatch_indirect_for_svo::<I>(context, svo_index, SLOT_PADDED);
}

pub(crate) fn record_prefix_sum_for_svo<P, D, I>(
    context: &VkSecondaryRecordContext,
    depth: u32,
    svo_index: usize,
) where
    P: ComputePipelineMarker,
    D: DescriptorSetMarker,
    I: BufferMarker<Lifetime = Persistent>,
    D::Lifetime: DescriptorSetResolver,
{
    context.cmd_bind_descriptor_set::<P, D>(0, &[]);

    let dispatch_phase = |phase: u32, slot: usize| {
        let push_constants = VoxelPushConstants {
            depth,
            layer_type: LAYER_TYPE_PREFIX_SUM,
            phase,
            svo_index: svo_index as u32,
        };

        context.cmd_push_constants::<P, _>(ShaderStageFlags::COMPUTE, 0, &push_constants);
        dispatch_indirect_for_svo::<I>(context, svo_index, slot);
        insert_memory_barrier_to_secondary_buffer(context);
    };

    dispatch_phase(0, SLOT_PADDED);
    dispatch_phase(1, SLOT_ONE);
    dispatch_phase(2, SLOT_PADDED);
}

pub(crate) fn record_compact_unique_pairs_for_svo<P, D, I>(
    context: &VkSecondaryRecordContext,
    depth: u32,
    svo_index: usize,
) where
    P: ComputePipelineMarker,
    D: DescriptorSetMarker,
    I: BufferMarker<Lifetime = Persistent>,
    D::Lifetime: DescriptorSetResolver,
{
    context.cmd_bind_descriptor_set::<P, D>(0, &[]);

    let push_constants = VoxelPushConstants {
        depth,
        layer_type: LAYER_TYPE_COMPACT,
        phase: 0,
        svo_index: svo_index as u32,
    };

    context.cmd_push_constants::<P, _>(ShaderStageFlags::COMPUTE, 0, &push_constants);
    dispatch_indirect_for_svo::<I>(context, svo_index, SLOT_PADDED);
}

pub(crate) fn record_batch_nodes_for_svo<P, D, I>(
    context: &VkSecondaryRecordContext,
    depth: u32,
    svo_index: usize,
    layer_type: u32,
) where
    P: ComputePipelineMarker,
    D: DescriptorSetMarker,
    I: BufferMarker<Lifetime = Persistent>,
    D::Lifetime: DescriptorSetResolver,
{
    context.cmd_bind_descriptor_set::<P, D>(0, &[]);

    let push_constants = VoxelPushConstants {
        depth,
        layer_type,
        phase: 0,
        svo_index: svo_index as u32,
    };

    context.cmd_push_constants::<P, _>(ShaderStageFlags::COMPUTE, 0, &push_constants);
    dispatch_indirect_for_svo::<I>(context, svo_index, SLOT_PADDED);
}

pub(crate) fn record_update_leaves_for_svo<P, D, I>(
    context: &VkSecondaryRecordContext,
    svo_index: usize,
) where
    P: ComputePipelineMarker,
    D: DescriptorSetMarker,
    I: BufferMarker<Lifetime = Persistent>,
    D::Lifetime: DescriptorSetResolver,
{
    context.cmd_bind_descriptor_set::<P, D>(0, &[]);

    let push_constants = VoxelPushConstants {
        depth: 0,
        layer_type: LAYER_TYPE_UPDATE_LEAVES,
        phase: 0,
        svo_index: svo_index as u32,
    };

    context.cmd_push_constants::<P, _>(ShaderStageFlags::COMPUTE, 0, &push_constants);
    dispatch_indirect_for_svo::<I>(context, svo_index, SLOT_UNPADDED);
}

fn dispatch_indirect_for_svo<I>(context: &VkSecondaryRecordContext, svo_index: usize, slot: usize)
where
    I: BufferMarker<Lifetime = Persistent>,
{
    let offset =
        ((svo_index * INDIRECT_DISPATCH_SLOTS_PER_SVO + slot) as u64) * DISPATCH_INDIRECT_STRIDE;
    context.cmd_dispatch_indirect::<I>(0, offset);
}
