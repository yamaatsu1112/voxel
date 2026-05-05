use crate::rendering::commands::{CreateComputePipelineCommand, CreateGraphicsPipelineCommand};
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct PipelineBuilder;

impl PipelineBuilder {
    pub fn build_renderer_pipelines(renderer: &mut Renderer<VulkanRenderer>) {
        use crate::builders::configs::pipeline_configs::*;

        renderer.add_command(CreateGraphicsPipelineCommand::<MainPipeline>::new());
        renderer.add_command(CreateGraphicsPipelineCommand::<GBufferWritePipeline>::new());
        renderer.add_command(CreateGraphicsPipelineCommand::<GBufferColorWritePipeline>::new());
        renderer.add_command(CreateComputePipelineCommand::<PrepareDispatchPipeline>::new());
        renderer.add_command(CreateGraphicsPipelineCommand::<DynamicGBufferWritePipeline>::new());
        renderer.add_command(CreateGraphicsPipelineCommand::<
            DynamicGBufferColorWritePipeline,
        >::new());
    }
}
