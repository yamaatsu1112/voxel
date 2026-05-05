use ash::vk;

use crate::voxel::svo::grouped::buffers::{
    DynamicIdSVOLeavesOriginalBuffer, DynamicIdSVOLeavesPerFrameBuffer,
    DynamicIdSVONodesOriginalBuffer, DynamicIdSVONodesPerFrameBuffer,
    DynamicSVOLeavesOriginalBuffer, DynamicSVOLeavesPerFrameBuffer, DynamicSVONodesOriginalBuffer,
    DynamicSVONodesPerFrameBuffer, ID_BIT_PLANES,
};
use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;

#[derive(Clone)]
pub struct CopyDynamicSVO;

impl VkGraphicsRecordable for CopyDynamicSVO {
    fn record(context: &mut VkGraphicsRecordContext) {
        let svo_node_buffer_size = context
            .get_buffer_size::<DynamicSVONodesPerFrameBuffer>(0)
            .expect("Failed to get dynamic SVO per-frame node buffer size");
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: svo_node_buffer_size,
        };
        context.cmd_copy_buffer::<DynamicSVONodesOriginalBuffer, DynamicSVONodesPerFrameBuffer>(
            0,
            0,
            &[copy_region],
        );

        let svo_leaf_buffer_size = context
            .get_buffer_size::<DynamicSVOLeavesPerFrameBuffer>(0)
            .expect("Failed to get dynamic SVO per-frame leaf buffer size");
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: svo_leaf_buffer_size,
        };
        context.cmd_copy_buffer::<DynamicSVOLeavesOriginalBuffer, DynamicSVOLeavesPerFrameBuffer>(
            0,
            0,
            &[copy_region],
        );

        for bit_index in 0..ID_BIT_PLANES {
            let id_svo_node_buffer_size = context
                .get_buffer_size::<DynamicIdSVONodesPerFrameBuffer>(bit_index)
                .expect("Failed to get dynamic ID SVO per-frame node buffer size");
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: id_svo_node_buffer_size,
            };
            context.cmd_copy_buffer::<DynamicIdSVONodesOriginalBuffer, DynamicIdSVONodesPerFrameBuffer>(
                bit_index,
                bit_index,
                &[copy_region],
            );

            let id_svo_leaf_buffer_size = context
                .get_buffer_size::<DynamicIdSVOLeavesPerFrameBuffer>(bit_index)
                .expect("Failed to get dynamic ID SVO per-frame leaf buffer size");
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: id_svo_leaf_buffer_size,
            };
            context
                .cmd_copy_buffer::<DynamicIdSVOLeavesOriginalBuffer, DynamicIdSVOLeavesPerFrameBuffer>(
                    bit_index,
                    bit_index,
                    &[copy_region],
                );
        }

        let memory_barrier = vk::MemoryBarrier {
            s_type: vk::StructureType::MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            _marker: std::marker::PhantomData,
        };
        context.cmd_pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[memory_barrier],
            &[],
            &[],
        );
    }
}
