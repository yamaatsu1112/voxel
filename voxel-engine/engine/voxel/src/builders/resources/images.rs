use ash::vk;

use crate::builders::configs::image_configs::{
    DynamicGBufferColorImage, DynamicGBufferColorImageView, DynamicGBufferDepthImage,
    DynamicGBufferDepthImageView, DynamicGBufferImage, DynamicGBufferImageView, GBufferColorImage,
    GBufferColorImageView, GBufferDepthImage, GBufferDepthImageView, GBufferImage,
    GBufferImageView,
};
use crate::recordables::GraphicsTransferMutex;
use crate::rendering::commands::{
    CreatePerFrameImageViewsCommand, CreatePerFrameImagesCommand, TransitionPerFrameLayoutCommand,
};
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub fn create_gbuffer_per_frame_images(renderer: &mut Renderer<VulkanRenderer>) {
    renderer.add_command(CreatePerFrameImagesCommand::<GBufferImage>::new());
    renderer.add_command(TransitionPerFrameLayoutCommand::<
        GraphicsTransferMutex,
        GBufferImage,
    >::new(
        0,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(CreatePerFrameImageViewsCommand::<GBufferImageView>::new());

    renderer.add_command(CreatePerFrameImagesCommand::<GBufferDepthImage>::new());
    renderer.add_command(TransitionPerFrameLayoutCommand::<
        GraphicsTransferMutex,
        GBufferDepthImage,
    >::new(
        0,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(CreatePerFrameImageViewsCommand::<GBufferDepthImageView>::new());

    renderer.add_command(CreatePerFrameImagesCommand::<GBufferColorImage>::new());
    renderer.add_command(TransitionPerFrameLayoutCommand::<
        GraphicsTransferMutex,
        GBufferColorImage,
    >::new(
        0,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(CreatePerFrameImageViewsCommand::<GBufferColorImageView>::new());

    renderer.add_command(CreatePerFrameImagesCommand::<DynamicGBufferImage>::new());
    renderer.add_command(TransitionPerFrameLayoutCommand::<
        GraphicsTransferMutex,
        DynamicGBufferImage,
    >::new(
        0,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(CreatePerFrameImageViewsCommand::<DynamicGBufferImageView>::new());

    renderer.add_command(CreatePerFrameImagesCommand::<DynamicGBufferDepthImage>::new());
    renderer.add_command(TransitionPerFrameLayoutCommand::<
        GraphicsTransferMutex,
        DynamicGBufferDepthImage,
    >::new(
        0,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(CreatePerFrameImageViewsCommand::<
        DynamicGBufferDepthImageView,
    >::new());

    renderer.add_command(CreatePerFrameImagesCommand::<DynamicGBufferColorImage>::new());
    renderer.add_command(TransitionPerFrameLayoutCommand::<
        GraphicsTransferMutex,
        DynamicGBufferColorImage,
    >::new(
        0,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(CreatePerFrameImageViewsCommand::<
        DynamicGBufferColorImageView,
    >::new());
}
