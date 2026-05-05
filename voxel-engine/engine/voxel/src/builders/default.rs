use crate::builders::configs::descriptor_set_configs::*;
use crate::builders::configs::image_configs::*;
use crate::builders::configs::sampler_configs;
use crate::builders::descriptor_builder::DescriptorBuilder;
use crate::builders::pipeline_builder::PipelineBuilder;
use crate::builders::resources::ResourceBuilder;
use crate::recordables::GraphicsTransferMutex;
use crate::rendering::commands::{
    BindPerFrameImageCommand, CreatePerFrameImageViewsCommand, CreatePerFrameImagesCommand,
    DestroyPerFrameImageViewsCommand, DestroyPerFrameImagesCommand, RegisterSwapchainResizeCommand,
    TransitionPerFrameLayoutCommand,
};
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

use ash::vk;

pub fn setup_voxel_renderer(renderer: &mut Renderer<VulkanRenderer>) {
    ResourceBuilder::build_renderer_resources(renderer);
    DescriptorBuilder::build_renderer_descriptors(renderer);
    crate::voxel::svo::svo_config::register_resources(renderer);
    register_swapchain_resize_commands(renderer);
    PipelineBuilder::build_renderer_pipelines(renderer);
}

fn register_swapchain_resize_commands(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImageViewsCommand::<GBufferImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImageViewsCommand::<GBufferDepthImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImageViewsCommand::<GBufferColorImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImageViewsCommand::<DynamicGBufferImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImageViewsCommand::<DynamicGBufferDepthImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImageViewsCommand::<DynamicGBufferColorImageView>::new(),
    )));

    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImagesCommand::<GBufferImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImagesCommand::<GBufferDepthImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImagesCommand::<GBufferColorImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImagesCommand::<DynamicGBufferImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImagesCommand::<DynamicGBufferDepthImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        DestroyPerFrameImagesCommand::<DynamicGBufferColorImage>::new(),
    )));

    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImagesCommand::<GBufferImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        TransitionPerFrameLayoutCommand::<GraphicsTransferMutex, GBufferImage>::new(
            0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        ),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImageViewsCommand::<GBufferImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImagesCommand::<GBufferDepthImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        TransitionPerFrameLayoutCommand::<GraphicsTransferMutex, GBufferDepthImage>::new(
            0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        ),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImageViewsCommand::<GBufferDepthImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImagesCommand::<GBufferColorImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        TransitionPerFrameLayoutCommand::<GraphicsTransferMutex, GBufferColorImage>::new(
            0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        ),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImageViewsCommand::<GBufferColorImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImagesCommand::<DynamicGBufferImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        TransitionPerFrameLayoutCommand::<GraphicsTransferMutex, DynamicGBufferImage>::new(
            0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        ),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImageViewsCommand::<DynamicGBufferImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImagesCommand::<DynamicGBufferDepthImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        TransitionPerFrameLayoutCommand::<GraphicsTransferMutex, DynamicGBufferDepthImage>::new(
            0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        ),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImageViewsCommand::<DynamicGBufferDepthImageView>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImagesCommand::<DynamicGBufferColorImage>::new(),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        TransitionPerFrameLayoutCommand::<GraphicsTransferMutex, DynamicGBufferColorImage>::new(
            0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        ),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        CreatePerFrameImageViewsCommand::<DynamicGBufferColorImageView>::new(),
    )));

    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        BindPerFrameImageCommand::<
            MainDescriptorSet,
            GBufferImageView,
            sampler_configs::GBufferSampler,
        >::new(3, 0, 0, 1),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        BindPerFrameImageCommand::<
            MainDescriptorSet,
            GBufferDepthImageView,
            sampler_configs::GBufferSampler,
        >::new(4, 0, 0, 1),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        BindPerFrameImageCommand::<
            MainDescriptorSet,
            GBufferColorImageView,
            sampler_configs::GBufferSampler,
        >::new(5, 0, 0, 1),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        BindPerFrameImageCommand::<
            MainDescriptorSet,
            DynamicGBufferImageView,
            sampler_configs::GBufferSampler,
        >::new(6, 0, 0, 1),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        BindPerFrameImageCommand::<
            MainDescriptorSet,
            DynamicGBufferDepthImageView,
            sampler_configs::GBufferSampler,
        >::new(7, 0, 0, 1),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        BindPerFrameImageCommand::<
            MainDescriptorSet,
            DynamicGBufferColorImageView,
            sampler_configs::GBufferSampler,
        >::new(8, 0, 0, 1),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        BindPerFrameImageCommand::<
            GBufferColorWriteDescriptorSet,
            GBufferImageView,
            sampler_configs::GBufferSampler,
        >::new(5, 0, 0, 1),
    )));
    renderer.add_command(RegisterSwapchainResizeCommand::new(Box::new(
        BindPerFrameImageCommand::<
            DynamicGBufferColorWriteDescriptorSet,
            DynamicGBufferImageView,
            sampler_configs::GBufferSampler,
        >::new(5, 0, 0, 1),
    )));
}
