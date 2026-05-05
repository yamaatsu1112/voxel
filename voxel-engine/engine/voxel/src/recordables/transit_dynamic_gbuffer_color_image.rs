use crate::builders::configs::image_configs::DynamicGBufferColorImage;
use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;
use ash::vk;
use std::marker::PhantomData;
use std::ptr;

#[derive(Clone)]
pub struct TransitDynamicGBufferColorImage;

impl VkGraphicsRecordable for TransitDynamicGBufferColorImage {
    fn record(context: &mut VkGraphicsRecordContext) {
        let gbuffer_color_image = context
            .get_per_frame_image::<DynamicGBufferColorImage>(0)
            .expect("Failed to get Dynamic G-Buffer color image");

        // Transition color G-Buffer from COLOR_ATTACHMENT_OPTIMAL to SHADER_READ_ONLY_OPTIMAL
        let barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image: gbuffer_color_image,
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
            &[barrier],
        );
    }
}
