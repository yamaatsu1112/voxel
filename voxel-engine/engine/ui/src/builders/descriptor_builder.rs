use crate::builders::configs::descriptor_set_configs::UIDescriptorSet;
use crate::builders::configs::sampler_configs::UITextureSampler;
use crate::rendering::commands::{CreateDescriptorSetsCommand, CreateSamplerCommand};
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct DescriptorBuilder;

impl DescriptorBuilder {
    pub fn build_renderer_descriptors(renderer: &mut Renderer<VulkanRenderer>) {
        renderer.add_command(CreateSamplerCommand::<UITextureSampler>::new());
        renderer.add_command(CreateDescriptorSetsCommand::<UIDescriptorSet>::new());
    }
}
