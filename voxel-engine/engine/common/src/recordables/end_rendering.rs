use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;

use ash::vk;
use std::marker::PhantomData;
use std::ptr;

#[derive(Clone)]
pub struct EndRenderingPhase;

impl VkGraphicsRecordable for EndRenderingPhase {
    fn record(context: &mut VkGraphicsRecordContext) {
        context.cmd_end_rendering();
    }
}

#[derive(Clone)]
pub struct EndRendering;

impl VkGraphicsRecordable for EndRendering {
    fn record(context: &mut VkGraphicsRecordContext) {
        let image = context.get_swapchain_image(context.image_index());

        context.cmd_end_rendering();

        let image_memory_barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::empty(),
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image,
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
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[image_memory_barrier],
        );
    }
}
