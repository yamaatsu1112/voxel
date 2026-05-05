use crate::builders::configs::buffer_configs::*;
use crate::builders::configs::descriptor_set_configs::UIDescriptorSet;
use crate::builders::configs::pipeline_configs::UIPipeline;
use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;

/// Count of UI instances for the current frame resource snapshot
#[derive(Clone, Copy)]
pub struct UIInstanceCount(pub u32);

#[derive(Clone)]
pub struct UIPipelineRecordable;

impl VkGraphicsRecordable for UIPipelineRecordable {
    fn record(context: &mut VkGraphicsRecordContext) {
        let instance_count = context
            .record_resource()
            .get::<UIInstanceCount>()
            .map(|count| count.0)
            .unwrap_or(0);

        context.cmd_bind_graphics_pipeline::<UIPipeline>();
        context.cmd_bind_descriptor_set::<UIPipeline, UIDescriptorSet>(0, &[]);
        context.cmd_bind_vertex_buffer::<UIInstanceBuffer>(0, 0, 0);
        context.cmd_draw(6, instance_count, 0, 0);
    }
}
