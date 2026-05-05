use ash::vk;
use engine_common::ComputeUniformBuffer;

use crate::builders::configs::buffer_configs::{
    CollisionInputBuffer, CollisionResultBuffer, CommandCountBuffer, DestroyCommandCountBuffer,
    DestroyLeafEditCommandBuffer, DynamicCommandCountBuffer, DynamicFlagBuffer,
    DynamicLeafEditCommandBuffer, DynamicPrefixSumBuffer, DynamicRequestCountBuffer, FlagBuffer,
    LeafEditCommandBuffer, PrefixSumBuffer, RaycastResultBuffer, RequestCountBuffer,
};
use crate::builders::configs::descriptor_set_configs::{
    DynamicGBufferColorWriteDescriptorSet, DynamicGBufferWriteDescriptorSet,
    GBufferColorWriteDescriptorSet, MainDescriptorSet,
};
use crate::builders::resources::buffers::{
    create_dynamic_id_svo_buffers, create_dynamic_storage_buffers, create_id_svo_buffers,
    create_storage_buffers,
};
use crate::rendering::commands::{
    BindBufferCommand, BindPerFrameBufferCommand, CreateBuffersCommand,
    CreateComputePipelineCommand, CreateDescriptorSetsCommand,
};
use crate::rendering::renderer::Renderer;
use crate::voxel::svo::sv64_individual::buffers::{
    AllocationRequestBuffer, DestroyIndirectDispatchBuffer, DynamicAllocationRequestBuffer,
    DynamicIdSVOLeavesOriginalBuffer, DynamicIdSVOLeavesPerFrameBuffer,
    DynamicIdSVONodesOriginalBuffer, DynamicIdSVONodesPerFrameBuffer,
    DynamicIndirectDispatchBuffer, DynamicSVOLeavesOriginalBuffer, DynamicSVOLeavesPerFrameBuffer,
    DynamicSVONodesOriginalBuffer, DynamicSVONodesPerFrameBuffer, DynamicUniqueRequestBuffer,
    ID_BIT_PLANES, IdSVOLeavesOriginalBuffer, IdSVOLeavesPerFrameBuffer, IdSVONodesOriginalBuffer,
    IdSVONodesPerFrameBuffer, IndirectDispatchBuffer, SVOLeavesOriginalBuffer,
    SVOLeavesPerFrameBuffer, SVONodesOriginalBuffer, SVONodesPerFrameBuffer, UniqueRequestBuffer,
};
use crate::voxel::svo::sv64_individual::descriptor_sets::{
    CollisionComputeDescriptorSet, DynamicVoxelDescriptorSet, PrepareDestroyDispatchDescriptorSet,
    PrepareDynamicDispatchDescriptorSet, PrepareVoxelDispatchDescriptorSet,
    RaycastComputeDescriptorSet, VoxelDescriptorSet, VoxelDestroyDescriptorSet,
};
use crate::voxel::svo::sv64_individual::pipelines::{
    CollisionComputePipeline, DynamicVoxelPipeline, RaycastComputePipeline, VoxelDestroyPipeline,
    VoxelPipeline,
};
use crate::vulkan::resource_config::{BufferMarker, DescriptorSetMarker};
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub(crate) fn register_resources(renderer: &mut Renderer<VulkanRenderer>) {
    create_storage_buffers(renderer);
    create_id_svo_buffers(renderer);
    create_dynamic_storage_buffers(renderer);
    create_dynamic_id_svo_buffers(renderer);
    create_indirect_dispatch_buffers(renderer);
    create_request_buffers(renderer);

    create_descriptor_sets(renderer);
    bind_main_descriptor_sets(renderer);
    bind_voxel_descriptor_sets(renderer);
    bind_destroy_descriptor_sets(renderer);
    bind_raycast_descriptor_sets(renderer);
    bind_collision_descriptor_sets(renderer);
    bind_gbuffer_color_write_descriptor_sets(renderer);
    bind_dynamic_voxel_descriptor_sets(renderer);
    bind_dynamic_gbuffer_write_descriptor_sets(renderer);
    bind_dynamic_gbuffer_color_write_descriptor_sets(renderer);
    bind_prepare_dispatch_descriptor_sets(renderer);

    create_pipelines(renderer);
}

fn create_indirect_dispatch_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<IndirectDispatchBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DestroyIndirectDispatchBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicIndirectDispatchBuffer>::new());
}

fn create_request_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<AllocationRequestBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<UniqueRequestBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicAllocationRequestBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicUniqueRequestBuffer>::new());
}

fn create_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateDescriptorSetsCommand::<VoxelDestroyDescriptorSet>::new());
    renderer.add_command(CreateDescriptorSetsCommand::<VoxelDescriptorSet>::new());
    renderer.add_command(CreateDescriptorSetsCommand::<RaycastComputeDescriptorSet>::new());
    renderer.add_command(CreateDescriptorSetsCommand::<DynamicVoxelDescriptorSet>::new());
    renderer.add_command(CreateDescriptorSetsCommand::<CollisionComputeDescriptorSet>::new());
    renderer.add_command(CreateDescriptorSetsCommand::<
        PrepareVoxelDispatchDescriptorSet,
    >::new());
    renderer.add_command(CreateDescriptorSetsCommand::<
        PrepareDestroyDispatchDescriptorSet,
    >::new());
    renderer.add_command(CreateDescriptorSetsCommand::<
        PrepareDynamicDispatchDescriptorSet,
    >::new());
}

fn create_pipelines(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateComputePipelineCommand::<VoxelDestroyPipeline>::new());
    renderer.add_command(CreateComputePipelineCommand::<VoxelPipeline>::new());
    renderer.add_command(CreateComputePipelineCommand::<RaycastComputePipeline>::new());
    renderer.add_command(CreateComputePipelineCommand::<CollisionComputePipeline>::new());
    renderer.add_command(CreateComputePipelineCommand::<DynamicVoxelPipeline>::new());
}

fn bind_main_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(BindPerFrameBufferCommand::<
        MainDescriptorSet,
        SVONodesPerFrameBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindPerFrameBufferCommand::<
        MainDescriptorSet,
        SVOLeavesPerFrameBuffer,
    >::new(2, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
}

fn bind_voxel_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;

    bind_mixed_svo_component_buffers::<
        VoxelDescriptorSet,
        SVONodesOriginalBuffer,
        IdSVONodesOriginalBuffer,
        SVOLeavesOriginalBuffer,
        IdSVOLeavesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::SVOCountersOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::IdSVOCountersOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::SVOFreeNodeIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::IdSVOFreeNodeIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::SVOFreeLeafIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::IdSVOFreeLeafIndicesOriginalBuffer,
    >(renderer, NUM_SVO_DATA);

    renderer.add_command(
        BindBufferCommand::<VoxelDescriptorSet, LeafEditCommandBuffer>::new(
            5,
            0,
            0,
            NUM_SVO_DATA,
            vk::DescriptorType::STORAGE_BUFFER,
        ),
    );
    renderer.add_command(BindBufferCommand::<
        VoxelDescriptorSet,
        AllocationRequestBuffer,
    >::new(
        6, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(
        BindBufferCommand::<VoxelDescriptorSet, UniqueRequestBuffer>::new(
            7,
            0,
            0,
            NUM_SVO_DATA,
            vk::DescriptorType::STORAGE_BUFFER,
        ),
    );
    renderer.add_command(BindBufferCommand::<VoxelDescriptorSet, FlagBuffer>::new(
        8,
        0,
        0,
        NUM_SVO_DATA,
        vk::DescriptorType::STORAGE_BUFFER,
    ));
    renderer.add_command(
        BindBufferCommand::<VoxelDescriptorSet, PrefixSumBuffer>::new(
            9,
            0,
            0,
            NUM_SVO_DATA,
            vk::DescriptorType::STORAGE_BUFFER,
        ),
    );
    renderer.add_command(
        BindBufferCommand::<VoxelDescriptorSet, RequestCountBuffer>::new(
            10,
            0,
            0,
            NUM_SVO_DATA,
            vk::DescriptorType::STORAGE_BUFFER,
        ),
    );
    renderer.add_command(
        BindBufferCommand::<VoxelDescriptorSet, ComputeUniformBuffer>::new(
            11,
            0,
            0,
            1,
            vk::DescriptorType::UNIFORM_BUFFER,
        ),
    );
    renderer.add_command(
        BindBufferCommand::<VoxelDescriptorSet, CommandCountBuffer>::new(
            12,
            0,
            0,
            1,
            vk::DescriptorType::STORAGE_BUFFER,
        ),
    );
}

fn bind_destroy_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;

    bind_mixed_svo_component_buffers::<
        VoxelDestroyDescriptorSet,
        SVONodesOriginalBuffer,
        IdSVONodesOriginalBuffer,
        SVOLeavesOriginalBuffer,
        IdSVOLeavesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::SVOCountersOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::IdSVOCountersOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::SVOFreeNodeIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::IdSVOFreeNodeIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::SVOFreeLeafIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::IdSVOFreeLeafIndicesOriginalBuffer,
    >(renderer, NUM_SVO_DATA);

    renderer.add_command(BindBufferCommand::<
        VoxelDestroyDescriptorSet,
        DestroyLeafEditCommandBuffer,
    >::new(
        5, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        VoxelDestroyDescriptorSet,
        AllocationRequestBuffer,
    >::new(
        6, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        VoxelDestroyDescriptorSet,
        UniqueRequestBuffer,
    >::new(
        7, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(
        BindBufferCommand::<VoxelDestroyDescriptorSet, FlagBuffer>::new(
            8,
            0,
            0,
            NUM_SVO_DATA,
            vk::DescriptorType::STORAGE_BUFFER,
        ),
    );
    renderer.add_command(BindBufferCommand::<
        VoxelDestroyDescriptorSet,
        PrefixSumBuffer,
    >::new(
        9, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        VoxelDestroyDescriptorSet,
        RequestCountBuffer,
    >::new(
        10, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        VoxelDestroyDescriptorSet,
        ComputeUniformBuffer,
    >::new(11, 0, 0, 1, vk::DescriptorType::UNIFORM_BUFFER));
    renderer.add_command(BindBufferCommand::<
        VoxelDestroyDescriptorSet,
        DestroyCommandCountBuffer,
    >::new(12, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
}

fn bind_raycast_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(BindBufferCommand::<
        RaycastComputeDescriptorSet,
        ComputeUniformBuffer,
    >::new(0, 0, 0, 1, vk::DescriptorType::UNIFORM_BUFFER));
    renderer.add_command(BindBufferCommand::<
        RaycastComputeDescriptorSet,
        SVONodesOriginalBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindBufferCommand::<
        RaycastComputeDescriptorSet,
        SVOLeavesOriginalBuffer,
    >::new(2, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindBufferCommand::<
        RaycastComputeDescriptorSet,
        RaycastResultBuffer,
    >::new(3, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
}

fn bind_collision_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(BindBufferCommand::<
        CollisionComputeDescriptorSet,
        CollisionInputBuffer,
    >::new(0, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindBufferCommand::<
        CollisionComputeDescriptorSet,
        SVONodesOriginalBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindBufferCommand::<
        CollisionComputeDescriptorSet,
        SVOLeavesOriginalBuffer,
    >::new(2, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindBufferCommand::<
        CollisionComputeDescriptorSet,
        CollisionResultBuffer,
    >::new(3, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
}

fn bind_gbuffer_color_write_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(BindPerFrameBufferCommand::<
        GBufferColorWriteDescriptorSet,
        SVONodesPerFrameBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindPerFrameBufferCommand::<
        GBufferColorWriteDescriptorSet,
        SVOLeavesPerFrameBuffer,
    >::new(2, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindPerFrameBufferCommand::<
        GBufferColorWriteDescriptorSet,
        IdSVONodesPerFrameBuffer,
    >::new(
        3, 0, 0, ID_BIT_PLANES, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindPerFrameBufferCommand::<
        GBufferColorWriteDescriptorSet,
        IdSVOLeavesPerFrameBuffer,
    >::new(
        4, 0, 0, ID_BIT_PLANES, vk::DescriptorType::STORAGE_BUFFER
    ));
}

fn bind_dynamic_voxel_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;

    bind_mixed_svo_component_buffers::<
        DynamicVoxelDescriptorSet,
        DynamicSVONodesOriginalBuffer,
        DynamicIdSVONodesOriginalBuffer,
        DynamicSVOLeavesOriginalBuffer,
        DynamicIdSVOLeavesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::DynamicSVOCountersOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::DynamicIdSVOCountersOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::DynamicSVOFreeNodeIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::DynamicIdSVOFreeNodeIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::DynamicSVOFreeLeafIndicesOriginalBuffer,
        crate::voxel::svo::sv64_individual::buffers::DynamicIdSVOFreeLeafIndicesOriginalBuffer,
    >(renderer, NUM_SVO_DATA);

    renderer.add_command(BindBufferCommand::<
        DynamicVoxelDescriptorSet,
        DynamicLeafEditCommandBuffer,
    >::new(
        5, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        DynamicVoxelDescriptorSet,
        DynamicAllocationRequestBuffer,
    >::new(
        6, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        DynamicVoxelDescriptorSet,
        DynamicUniqueRequestBuffer,
    >::new(
        7, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        DynamicVoxelDescriptorSet,
        DynamicFlagBuffer,
    >::new(
        8, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        DynamicVoxelDescriptorSet,
        DynamicPrefixSumBuffer,
    >::new(
        9, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        DynamicVoxelDescriptorSet,
        DynamicRequestCountBuffer,
    >::new(
        10, 0, 0, NUM_SVO_DATA, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindBufferCommand::<
        DynamicVoxelDescriptorSet,
        ComputeUniformBuffer,
    >::new(11, 0, 0, 1, vk::DescriptorType::UNIFORM_BUFFER));
    renderer.add_command(BindBufferCommand::<
        DynamicVoxelDescriptorSet,
        DynamicCommandCountBuffer,
    >::new(12, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
}

fn bind_dynamic_gbuffer_write_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(BindPerFrameBufferCommand::<
        DynamicGBufferWriteDescriptorSet,
        DynamicSVONodesPerFrameBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindPerFrameBufferCommand::<
        DynamicGBufferWriteDescriptorSet,
        DynamicSVOLeavesPerFrameBuffer,
    >::new(2, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
}

fn bind_dynamic_gbuffer_color_write_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(BindPerFrameBufferCommand::<
        DynamicGBufferColorWriteDescriptorSet,
        DynamicSVONodesPerFrameBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindPerFrameBufferCommand::<
        DynamicGBufferColorWriteDescriptorSet,
        DynamicSVOLeavesPerFrameBuffer,
    >::new(2, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindPerFrameBufferCommand::<
        DynamicGBufferColorWriteDescriptorSet,
        DynamicIdSVONodesPerFrameBuffer,
    >::new(
        3, 0, 0, ID_BIT_PLANES, vk::DescriptorType::STORAGE_BUFFER
    ));
    renderer.add_command(BindPerFrameBufferCommand::<
        DynamicGBufferColorWriteDescriptorSet,
        DynamicIdSVOLeavesPerFrameBuffer,
    >::new(
        4, 0, 0, ID_BIT_PLANES, vk::DescriptorType::STORAGE_BUFFER
    ));
}

fn bind_prepare_dispatch_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(BindBufferCommand::<
        PrepareVoxelDispatchDescriptorSet,
        CommandCountBuffer,
    >::new(0, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindBufferCommand::<
        PrepareVoxelDispatchDescriptorSet,
        IndirectDispatchBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));

    renderer.add_command(BindBufferCommand::<
        PrepareDestroyDispatchDescriptorSet,
        DestroyCommandCountBuffer,
    >::new(0, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindBufferCommand::<
        PrepareDestroyDispatchDescriptorSet,
        DestroyIndirectDispatchBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));

    renderer.add_command(BindBufferCommand::<
        PrepareDynamicDispatchDescriptorSet,
        DynamicCommandCountBuffer,
    >::new(0, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
    renderer.add_command(BindBufferCommand::<
        PrepareDynamicDispatchDescriptorSet,
        DynamicIndirectDispatchBuffer,
    >::new(1, 0, 0, 1, vk::DescriptorType::STORAGE_BUFFER));
}

fn bind_mixed_svo_component_buffers<
    TDescriptorSet,
    TSvoNodes,
    TIdSvoNodes,
    TSvoLeaves,
    TIdSvoLeaves,
    TSvoCounters,
    TIdSvoCounters,
    TSvoFreeNodeIndices,
    TIdSvoFreeNodeIndices,
    TSvoFreeLeafIndices,
    TIdSvoFreeLeafIndices,
>(
    renderer: &mut Renderer<VulkanRenderer>,
    num_svo_data: usize,
) where
    TDescriptorSet: DescriptorSetMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TSvoNodes: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TIdSvoNodes: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TSvoLeaves: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TIdSvoLeaves: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TSvoCounters: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TIdSvoCounters: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TSvoFreeNodeIndices: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TIdSvoFreeNodeIndices: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TSvoFreeLeafIndices: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TIdSvoFreeLeafIndices: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
{
    bind_mixed_svo_component_buffer::<TDescriptorSet, TSvoNodes, TIdSvoNodes>(
        renderer,
        0,
        num_svo_data,
    );
    bind_mixed_svo_component_buffer::<TDescriptorSet, TSvoLeaves, TIdSvoLeaves>(
        renderer,
        1,
        num_svo_data,
    );
    bind_mixed_svo_component_buffer::<TDescriptorSet, TSvoCounters, TIdSvoCounters>(
        renderer,
        2,
        num_svo_data,
    );
    bind_mixed_svo_component_buffer::<TDescriptorSet, TSvoFreeNodeIndices, TIdSvoFreeNodeIndices>(
        renderer,
        3,
        num_svo_data,
    );
    bind_mixed_svo_component_buffer::<TDescriptorSet, TSvoFreeLeafIndices, TIdSvoFreeLeafIndices>(
        renderer,
        4,
        num_svo_data,
    );
}

fn bind_mixed_svo_component_buffer<TDescriptorSet, TSvoBuffer, TIdSvoBuffer>(
    renderer: &mut Renderer<VulkanRenderer>,
    binding: u32,
    num_svo_data: usize,
) where
    TDescriptorSet: DescriptorSetMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TSvoBuffer: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
    TIdSvoBuffer: BufferMarker<Lifetime = crate::vulkan::resource_lifetime::Persistent>
        + Send
        + Sync
        + 'static,
{
    renderer.add_command(BindBufferCommand::<TDescriptorSet, TSvoBuffer>::new(
        binding,
        0,
        0,
        1,
        vk::DescriptorType::STORAGE_BUFFER,
    ));
    renderer.add_command(BindBufferCommand::<TDescriptorSet, TIdSvoBuffer>::new(
        binding,
        1,
        0,
        num_svo_data - 1,
        vk::DescriptorType::STORAGE_BUFFER,
    ));
}
