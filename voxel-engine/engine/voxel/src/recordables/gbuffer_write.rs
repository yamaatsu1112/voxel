use crate::builders::configs::buffer_configs::*;
use crate::builders::configs::descriptor_set_configs::MainDescriptorSet;
use crate::builders::configs::pipeline_configs::GBufferWritePipeline;
use crate::vulkan::record_context::{IndexType, VkGraphicsRecordContext};
use crate::vulkan::recordable::VkGraphicsRecordable;

#[derive(Clone)]
pub struct GBufferWritePipelineRecordable;

impl VkGraphicsRecordable for GBufferWritePipelineRecordable {
    fn record(context: &mut VkGraphicsRecordContext) {
        context.cmd_begin_timing("GBufferWritePipelineRecordable");
        context.cmd_bind_graphics_pipeline::<GBufferWritePipeline>();
        context.cmd_bind_descriptor_set::<GBufferWritePipeline, MainDescriptorSet>(0, &[]);
        context.cmd_bind_vertex_buffer::<FullscreenVertexBuffer>(0, 0, 0);
        context.cmd_bind_index_buffer::<IndexBuffer>(0, 0, IndexType::Uint32);

        use crate::vertex::INDICES;
        context.cmd_draw_indexed(INDICES.len() as u32, 1, 0, 0, 0);
        context.cmd_end_timing("GBufferWritePipelineRecordable");
    }
}
