use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;

use ash::vk;
use std::marker::PhantomData;
use std::ptr;

/// Begin rendering to swapchain image
#[derive(Clone)]
pub struct BeginSwapchainRendering {}

impl VkGraphicsRecordable for BeginSwapchainRendering {
    fn record(context: &mut VkGraphicsRecordContext) {
        let (width, height) = context.get_swapchain_extent();
        let extent = vk::Extent2D { width, height };
        let image_view = context.get_swapchain_image_view(context.image_index());
        let image = context.get_swapchain_image(context.image_index());

        let clear_color = [0.0, 0.0, 0.0, 1.0];
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear_color,
            },
        };

        let attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: ptr::null(),
            image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            clear_value,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            _marker: PhantomData,
        };

        let attachments = [attachment_info];
        let barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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

        let rendering_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            p_next: ptr::null(),
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            },
            flags: vk::RenderingFlagsKHR::empty(),
            layer_count: 1,
            view_mask: 0,
            color_attachment_count: attachments.len() as u32,
            p_color_attachments: attachments.as_ptr(),
            p_depth_attachment: ptr::null(),
            p_stencil_attachment: ptr::null(),
            _marker: PhantomData,
        };

        context.cmd_pipeline_barrier(
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
        context.cmd_begin_rendering(&rendering_info);
    }
}
