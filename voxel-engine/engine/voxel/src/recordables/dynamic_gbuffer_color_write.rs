use crate::builders::configs::buffer_configs::*;
use crate::builders::configs::descriptor_set_configs::DynamicGBufferColorWriteDescriptorSet;
use crate::builders::configs::pipeline_configs::DynamicGBufferColorWritePipeline;
use crate::builders::configs::pipeline_configs::gbuffer_color_write::ColorBitPushConstants;
use crate::vulkan::record_context::{IndexType, ShaderStageFlags, VkGraphicsRecordContext};
use crate::vulkan::recordable::VkGraphicsRecordable;

#[derive(Clone)]
pub struct DynamicGBufferColorWritePipelineRecordable;

impl VkGraphicsRecordable for DynamicGBufferColorWritePipelineRecordable {
    fn record(context: &mut VkGraphicsRecordContext) {
        context.cmd_bind_graphics_pipeline::<DynamicGBufferColorWritePipeline>();
        context.cmd_bind_descriptor_set::<
            DynamicGBufferColorWritePipeline,
            DynamicGBufferColorWriteDescriptorSet,
        >(0, &[]);
        context.cmd_bind_vertex_buffer::<FullscreenVertexBuffer>(0, 0, 0);
        context.cmd_bind_index_buffer::<IndexBuffer>(0, 0, IndexType::Uint32);

        use crate::vertex::INDICES;
        for bit_index in 0..8u32 {
            let push_constants = ColorBitPushConstants { bit_index };
            context.cmd_push_constants::<DynamicGBufferColorWritePipeline, _>(
                ShaderStageFlags::FRAGMENT,
                0,
                &push_constants,
            );
            context.cmd_draw_indexed(INDICES.len() as u32, 1, 0, 0, 0);
        }
    }
}
