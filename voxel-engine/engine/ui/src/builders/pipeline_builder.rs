use crate::rendering::commands::CreateGraphicsPipelineCommand;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct PipelineBuilder;

impl PipelineBuilder {
    pub fn build_renderer_pipelines(renderer: &mut Renderer<VulkanRenderer>) {
        use crate::builders::configs::pipeline_configs::UIPipeline;

        renderer.add_command(CreateGraphicsPipelineCommand::<UIPipeline>::new());
    }
}
