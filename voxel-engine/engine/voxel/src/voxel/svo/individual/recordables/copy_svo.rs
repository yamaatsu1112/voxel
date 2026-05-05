use ash::vk;

use crate::voxel::svo::individual::buffers::{
    ID_BIT_PLANES, IdSVOLeavesOriginalBuffer, IdSVOLeavesPerFrameBuffer, IdSVONodesOriginalBuffer,
    IdSVONodesPerFrameBuffer, SVOLeavesOriginalBuffer, SVOLeavesPerFrameBuffer,
    SVONodesOriginalBuffer, SVONodesPerFrameBuffer,
};
use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;

#[derive(Clone)]
pub struct CopySVO;

impl VkGraphicsRecordable for CopySVO {
    fn record(context: &mut VkGraphicsRecordContext) {
        let svo_node_buffer_size = context
            .get_buffer_size::<SVONodesPerFrameBuffer>(0)
            .expect("Failed to get SVO per-frame node buffer size");
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: svo_node_buffer_size,
        };
        context.cmd_copy_buffer::<SVONodesOriginalBuffer, SVONodesPerFrameBuffer>(
            0,
            0,
            &[copy_region],
        );

        let svo_leaf_buffer_size = context
            .get_buffer_size::<SVOLeavesPerFrameBuffer>(0)
            .expect("Failed to get SVO per-frame leaf buffer size");
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: svo_leaf_buffer_size,
        };
        context.cmd_copy_buffer::<SVOLeavesOriginalBuffer, SVOLeavesPerFrameBuffer>(
            0,
            0,
            &[copy_region],
        );

        for bit_index in 0..ID_BIT_PLANES {
            let id_svo_node_buffer_size = context
                .get_buffer_size::<IdSVONodesPerFrameBuffer>(bit_index)
                .expect("Failed to get ID SVO per-frame node buffer size");
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: id_svo_node_buffer_size,
            };
            context.cmd_copy_buffer::<IdSVONodesOriginalBuffer, IdSVONodesPerFrameBuffer>(
                bit_index,
                bit_index,
                &[copy_region],
            );

            let id_svo_leaf_buffer_size = context
                .get_buffer_size::<IdSVOLeavesPerFrameBuffer>(bit_index)
                .expect("Failed to get ID SVO per-frame leaf buffer size");
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: id_svo_leaf_buffer_size,
            };
            context.cmd_copy_buffer::<IdSVOLeavesOriginalBuffer, IdSVOLeavesPerFrameBuffer>(
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
