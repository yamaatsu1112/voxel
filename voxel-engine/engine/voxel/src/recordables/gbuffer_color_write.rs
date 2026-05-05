use crate::builders::configs::buffer_configs::*;
use crate::builders::configs::descriptor_set_configs::GBufferColorWriteDescriptorSet;
use crate::builders::configs::pipeline_configs::GBufferColorWritePipeline;
use crate::builders::configs::pipeline_configs::gbuffer_color_write::ColorBitPushConstants;
use crate::vulkan::record_context::{IndexType, ShaderStageFlags, VkGraphicsRecordContext};
use crate::vulkan::recordable::VkGraphicsRecordable;

#[derive(Clone)]
pub struct GBufferColorWritePipelineRecordable;

impl VkGraphicsRecordable for GBufferColorWritePipelineRecordable {
    fn record(context: &mut VkGraphicsRecordContext) {
        context.cmd_begin_timing("GBufferColorWritePipelineRecordable");
        context.cmd_bind_graphics_pipeline::<GBufferColorWritePipeline>();
        context
            .cmd_bind_descriptor_set::<GBufferColorWritePipeline, GBufferColorWriteDescriptorSet>(
                0,
                &[],
            );
        context.cmd_bind_vertex_buffer::<FullscreenVertexBuffer>(0, 0, 0);
        context.cmd_bind_index_buffer::<IndexBuffer>(0, 0, IndexType::Uint32);

        use crate::vertex::INDICES;
        for bit_index in 0..8u32 {
            let push_constants = ColorBitPushConstants { bit_index };
            context.cmd_push_constants::<GBufferColorWritePipeline, _>(
                ShaderStageFlags::FRAGMENT,
                0,
                &push_constants,
            );
            context.cmd_draw_indexed(INDICES.len() as u32, 1, 0, 0, 0);
        }
        context.cmd_end_timing("GBufferColorWritePipelineRecordable");
    }
}
