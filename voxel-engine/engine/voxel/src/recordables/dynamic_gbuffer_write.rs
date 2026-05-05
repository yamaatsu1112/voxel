use crate::builders::configs::buffer_configs::*;
use crate::builders::configs::descriptor_set_configs::DynamicGBufferWriteDescriptorSet;
use crate::builders::configs::pipeline_configs::DynamicGBufferWritePipeline;
use crate::vulkan::record_context::{IndexType, VkGraphicsRecordContext};
use crate::vulkan::recordable::VkGraphicsRecordable;

#[derive(Clone)]
pub struct DynamicGBufferWritePipelineRecordable;

impl VkGraphicsRecordable for DynamicGBufferWritePipelineRecordable {
    fn record(context: &mut VkGraphicsRecordContext) {
        context.cmd_bind_graphics_pipeline::<DynamicGBufferWritePipeline>();
        context.cmd_bind_descriptor_set::<DynamicGBufferWritePipeline, DynamicGBufferWriteDescriptorSet>(
            0,
            &[],
        );
        context.cmd_bind_vertex_buffer::<FullscreenVertexBuffer>(0, 0, 0);
        context.cmd_bind_index_buffer::<IndexBuffer>(0, 0, IndexType::Uint32);

        use crate::vertex::INDICES;
        context.cmd_draw_indexed(INDICES.len() as u32, 1, 0, 0, 0);
    }
}
