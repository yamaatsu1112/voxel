use crate::builders::configs::image_configs::{GBufferDepthImage, GBufferImage};
use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;
use ash::vk;
use std::marker::PhantomData;
use std::ptr;

#[derive(Clone)]
pub struct TransitGBufferImage;

impl VkGraphicsRecordable for TransitGBufferImage {
    fn record(context: &mut VkGraphicsRecordContext) {
        let gbuffer_image = context
            .get_per_frame_image::<GBufferImage>(0)
            .expect("Failed to get G-Buffer image");
        let gbuffer_depth_image = context
            .get_per_frame_image::<GBufferDepthImage>(0)
            .expect("Failed to get G-Buffer depth image");

        // transition gbuffer image from color attachment optimal layout to shader read only optimal layout
        let barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image: gbuffer_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: PhantomData,
        };

        let depth_barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image: gbuffer_depth_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: PhantomData,
        };

        context.cmd_pipeline_barrier(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier, depth_barrier],
        );
    }
}
