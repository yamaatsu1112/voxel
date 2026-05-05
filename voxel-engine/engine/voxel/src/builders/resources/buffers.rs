use crate::builders::configs::buffer_configs::compute::ID_BIT_PLANES;
use crate::builders::configs::buffer_configs::*;
use crate::recordables::GraphicsTransferMutex;
use crate::rendering::commands::{CreateBuffersCommand, UploadToBufferCommand};
use crate::rendering::renderer::Renderer;
use crate::vertex::{FULLSCREEN_VERTICES, INDICES};
use crate::voxel::svo::{SVO, SVOBackend};
use crate::vulkan::vulkan_renderer::VulkanRenderer;
use std::thread;

fn upload_split_svo<
    TNodeBuffer,
    TLeafBuffer,
    TCounterBuffer,
    TFreeNodeBuffer,
    TFreeLeafBuffer,
    TSvo,
>(
    renderer: &mut Renderer<VulkanRenderer>,
    index: usize,
    svo: &TSvo,
) where
    TNodeBuffer: crate::vulkan::resource_config::BufferMarker<
            Lifetime = crate::vulkan::resource_lifetime::Persistent,
        > + Send
        + Sync
        + 'static,
    TLeafBuffer: crate::vulkan::resource_config::BufferMarker<
            Lifetime = crate::vulkan::resource_lifetime::Persistent,
        > + Send
        + Sync
        + 'static,
    TCounterBuffer: crate::vulkan::resource_config::BufferMarker<
            Lifetime = crate::vulkan::resource_lifetime::Persistent,
        > + Send
        + Sync
        + 'static,
    TFreeNodeBuffer: crate::vulkan::resource_config::BufferMarker<
            Lifetime = crate::vulkan::resource_lifetime::Persistent,
        > + Send
        + Sync
        + 'static,
    TFreeLeafBuffer: crate::vulkan::resource_config::BufferMarker<
            Lifetime = crate::vulkan::resource_lifetime::Persistent,
        > + Send
        + Sync
        + 'static,
    TSvo: SVOBackend,
{
    renderer.add_command(UploadToBufferCommand::<
        GraphicsTransferMutex,
        TNodeBuffer,
        u8,
    >::new(index, svo.nodes_as_bytes(), 0));
    renderer.add_command(UploadToBufferCommand::<
        GraphicsTransferMutex,
        TLeafBuffer,
        u8,
    >::new(index, svo.leaves_as_bytes(), 0));
    renderer.add_command(UploadToBufferCommand::<
        GraphicsTransferMutex,
        TCounterBuffer,
        _,
    >::new(index, std::slice::from_ref(svo.counters()), 0));
    renderer.add_command(UploadToBufferCommand::<
        GraphicsTransferMutex,
        TFreeNodeBuffer,
        u8,
    >::new(index, svo.free_node_indices_as_bytes(), 0));
    renderer.add_command(UploadToBufferCommand::<
        GraphicsTransferMutex,
        TFreeLeafBuffer,
        u8,
    >::new(index, svo.free_leaf_indices_as_bytes(), 0));
}

pub fn create_fullscreen_vertex_buffer(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<FullscreenVertexBuffer>::new());
    renderer.add_command(UploadToBufferCommand::<
        GraphicsTransferMutex,
        FullscreenVertexBuffer,
        _,
    >::new(0, FULLSCREEN_VERTICES, 0));
}

pub fn create_index_buffer(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<IndexBuffer>::new());
    renderer.add_command(
        UploadToBufferCommand::<GraphicsTransferMutex, IndexBuffer, _>::new(0, INDICES, 0),
    );
}

pub fn create_storage_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    let empty_svo = thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(SVO::default)
        .expect("Failed to spawn thread for SVO initialization")
        .join()
        .expect("Failed to initialize empty SVO");
    renderer.add_command(CreateBuffersCommand::<SVONodesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<SVONodesPerFrameBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<SVOLeavesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<SVOLeavesPerFrameBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<SVOCountersOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<SVOFreeNodeIndicesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<SVOFreeLeafIndicesOriginalBuffer>::new());
    upload_split_svo::<
        SVONodesOriginalBuffer,
        SVOLeavesOriginalBuffer,
        SVOCountersOriginalBuffer,
        SVOFreeNodeIndicesOriginalBuffer,
        SVOFreeLeafIndicesOriginalBuffer,
        SVO,
    >(renderer, 0, &empty_svo);
}

pub fn create_leaf_edit_command_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<LeafEditCommandBuffer>::new());
}

pub fn create_destroy_leaf_edit_command_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<DestroyLeafEditCommandBuffer>::new());
}

pub fn create_command_count_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<CommandCountBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DestroyCommandCountBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicCommandCountBuffer>::new());
}

pub fn create_temporary_compute_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<FlagBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<PrefixSumBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<RequestCountBuffer>::new());
}

pub fn create_raycast_result_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<RaycastResultBuffer>::new());
}

pub fn create_collision_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<CollisionInputBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<CollisionResultBuffer>::new());
}

pub fn create_dynamic_storage_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    let empty_svo = thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(SVO::default)
        .expect("Failed to spawn thread for dynamic SVO initialization")
        .join()
        .expect("Failed to initialize empty dynamic SVO");
    renderer.add_command(CreateBuffersCommand::<DynamicSVONodesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicSVONodesPerFrameBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicSVOLeavesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicSVOLeavesPerFrameBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicSVOCountersOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<
        DynamicSVOFreeNodeIndicesOriginalBuffer,
    >::new());
    renderer.add_command(CreateBuffersCommand::<
        DynamicSVOFreeLeafIndicesOriginalBuffer,
    >::new());
    upload_split_svo::<
        DynamicSVONodesOriginalBuffer,
        DynamicSVOLeavesOriginalBuffer,
        DynamicSVOCountersOriginalBuffer,
        DynamicSVOFreeNodeIndicesOriginalBuffer,
        DynamicSVOFreeLeafIndicesOriginalBuffer,
        SVO,
    >(renderer, 0, &empty_svo);
}

pub fn create_id_svo_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    let empty_svo = thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(SVO::default)
        .expect("Failed to spawn thread for ID SVO initialization")
        .join()
        .expect("Failed to initialize empty ID SVO");
    renderer.add_command(CreateBuffersCommand::<IdSVONodesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<IdSVONodesPerFrameBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<IdSVOLeavesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<IdSVOLeavesPerFrameBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<IdSVOCountersOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<IdSVOFreeNodeIndicesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<IdSVOFreeLeafIndicesOriginalBuffer>::new());
    for bit_index in 0..ID_BIT_PLANES {
        upload_split_svo::<
            IdSVONodesOriginalBuffer,
            IdSVOLeavesOriginalBuffer,
            IdSVOCountersOriginalBuffer,
            IdSVOFreeNodeIndicesOriginalBuffer,
            IdSVOFreeLeafIndicesOriginalBuffer,
            SVO,
        >(renderer, bit_index, &empty_svo);
    }
}

pub fn create_dynamic_id_svo_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    let empty_svo = thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(SVO::default)
        .expect("Failed to spawn thread for dynamic ID SVO initialization")
        .join()
        .expect("Failed to initialize empty dynamic ID SVO");
    renderer.add_command(CreateBuffersCommand::<DynamicIdSVONodesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicIdSVONodesPerFrameBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicIdSVOLeavesOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicIdSVOLeavesPerFrameBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicIdSVOCountersOriginalBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<
        DynamicIdSVOFreeNodeIndicesOriginalBuffer,
    >::new());
    renderer.add_command(CreateBuffersCommand::<
        DynamicIdSVOFreeLeafIndicesOriginalBuffer,
    >::new());
    for bit_index in 0..ID_BIT_PLANES {
        upload_split_svo::<
            DynamicIdSVONodesOriginalBuffer,
            DynamicIdSVOLeavesOriginalBuffer,
            DynamicIdSVOCountersOriginalBuffer,
            DynamicIdSVOFreeNodeIndicesOriginalBuffer,
            DynamicIdSVOFreeLeafIndicesOriginalBuffer,
            SVO,
        >(renderer, bit_index, &empty_svo);
    }
}

pub fn create_dynamic_leaf_edit_command_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<DynamicLeafEditCommandBuffer>::new());
}

pub fn create_dynamic_temporary_compute_buffers(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreateBuffersCommand::<DynamicFlagBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicPrefixSumBuffer>::new());
    renderer.add_command(CreateBuffersCommand::<DynamicRequestCountBuffer>::new());
}
