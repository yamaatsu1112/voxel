use crate::builders::descriptor_builder::DescriptorBuilder;
use crate::builders::pipeline_builder::PipelineBuilder;
use crate::builders::resources::ResourceBuilder;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub fn setup_ui_renderer(renderer: &mut Renderer<VulkanRenderer>) {
    ResourceBuilder::build_renderer_resources(renderer);
    DescriptorBuilder::build_renderer_descriptors(renderer);
    PipelineBuilder::build_renderer_pipelines(renderer);
}
