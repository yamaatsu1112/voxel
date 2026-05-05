use crate::builders::configs::buffer_configs::*;
use crate::builders::configs::descriptor_set_configs::{
    MainDescriptorSet, MainStaticDescriptorSet,
};
use crate::builders::configs::pipeline_configs::MainPipeline;
use crate::vulkan::record_context::{IndexType, VkGraphicsRecordContext};
use crate::vulkan::recordable::VkGraphicsRecordable;

#[derive(Clone)]
pub struct MainPipelineRecordable;

impl VkGraphicsRecordable for MainPipelineRecordable {
    fn record(context: &mut VkGraphicsRecordContext) {
        context.cmd_begin_timing("MainPipelineRecordable");
        context.cmd_bind_graphics_pipeline::<MainPipeline>();
        context.cmd_bind_descriptor_set::<MainPipeline, MainDescriptorSet>(0, &[]);
        context.cmd_bind_descriptor_set::<MainPipeline, MainStaticDescriptorSet>(1, &[]);
        context.cmd_bind_vertex_buffer::<FullscreenVertexBuffer>(0, 0, 0);
        context.cmd_bind_index_buffer::<IndexBuffer>(0, 0, IndexType::Uint32);

        context.cmd_draw_indexed(crate::vertex::INDICES.len() as u32, 1, 0, 0, 0);
        context.cmd_end_timing("MainPipelineRecordable");
    }
}
