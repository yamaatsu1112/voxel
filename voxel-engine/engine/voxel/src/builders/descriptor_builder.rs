use ash::vk;
use engine_common::UniformBuffer;

use crate::builders::configs::descriptor_set_configs::*;
use crate::builders::configs::image_configs::*;
use crate::builders::configs::sampler_configs::*;
use crate::rendering::commands::*;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct DescriptorBuilder;

impl DescriptorBuilder {
    pub fn build_renderer_descriptors(renderer: &mut Renderer<VulkanRenderer>) {
        renderer.add_command(CreateSamplerCommand::<GBufferSampler>::new());
        renderer.add_command(CreateSamplerCommand::<VoxelTexture3DSampler>::new());

        renderer.add_command(CreateDescriptorSetsCommand::<MainDescriptorSet>::new());
        renderer.add_command(CreateDescriptorSetsCommand::<MainStaticDescriptorSet>::new());
        renderer.add_command(CreateDescriptorSetsCommand::<GBufferColorWriteDescriptorSet>::new());
        renderer.add_command(CreateDescriptorSetsCommand::<
            DynamicGBufferWriteDescriptorSet,
        >::new());
        renderer.add_command(CreateDescriptorSetsCommand::<
            DynamicGBufferColorWriteDescriptorSet,
        >::new());

        Self::bind_common_main_descriptor_sets(renderer);
        Self::bind_gbuffer_color_write_descriptor_sets(renderer);
        Self::bind_dynamic_gbuffer_write_descriptor_sets(renderer);
        Self::bind_dynamic_gbuffer_color_write_descriptor_sets(renderer);
        Self::bind_gbuffer_per_frame_images(renderer);
    }

    fn bind_common_main_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
        renderer.add_command(
            BindPerFrameBufferCommand::<MainDescriptorSet, UniformBuffer>::new(
                0,
                0,
                0,
                1,
                vk::DescriptorType::UNIFORM_BUFFER,
            ),
        );
    }

    fn bind_gbuffer_per_frame_images(renderer: &mut Renderer<VulkanRenderer>) {
        renderer.add_command(BindPerFrameImageCommand::<
            MainDescriptorSet,
            GBufferImageView,
            GBufferSampler,
        >::new(3, 0, 0, 1));
        renderer.add_command(BindPerFrameImageCommand::<
            MainDescriptorSet,
            GBufferDepthImageView,
            GBufferSampler,
        >::new(4, 0, 0, 1));
        renderer.add_command(BindPerFrameImageCommand::<
            MainDescriptorSet,
            GBufferColorImageView,
            GBufferSampler,
        >::new(5, 0, 0, 1));
        renderer.add_command(BindPerFrameImageCommand::<
            MainDescriptorSet,
            DynamicGBufferImageView,
            GBufferSampler,
        >::new(6, 0, 0, 1));
        renderer.add_command(BindPerFrameImageCommand::<
            MainDescriptorSet,
            DynamicGBufferDepthImageView,
            GBufferSampler,
        >::new(7, 0, 0, 1));
        renderer.add_command(BindPerFrameImageCommand::<
            MainDescriptorSet,
            DynamicGBufferColorImageView,
            GBufferSampler,
        >::new(8, 0, 0, 1));
        renderer.add_command(BindPerFrameImageCommand::<
            GBufferColorWriteDescriptorSet,
            GBufferImageView,
            GBufferSampler,
        >::new(5, 0, 0, 1));
        renderer.add_command(BindPerFrameImageCommand::<
            DynamicGBufferColorWriteDescriptorSet,
            DynamicGBufferImageView,
            GBufferSampler,
        >::new(5, 0, 0, 1));
    }

    fn bind_gbuffer_color_write_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
        renderer.add_command(BindPerFrameBufferCommand::<
            GBufferColorWriteDescriptorSet,
            UniformBuffer,
        >::new(
            0, 0, 0, 1, vk::DescriptorType::UNIFORM_BUFFER
        ));
    }

    fn bind_dynamic_gbuffer_write_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
        renderer.add_command(BindPerFrameBufferCommand::<
            DynamicGBufferWriteDescriptorSet,
            UniformBuffer,
        >::new(
            0, 0, 0, 1, vk::DescriptorType::UNIFORM_BUFFER
        ));
    }

    fn bind_dynamic_gbuffer_color_write_descriptor_sets(renderer: &mut Renderer<VulkanRenderer>) {
        renderer.add_command(BindPerFrameBufferCommand::<
            DynamicGBufferColorWriteDescriptorSet,
            UniformBuffer,
        >::new(
            0, 0, 0, 1, vk::DescriptorType::UNIFORM_BUFFER
        ));
    }
}
