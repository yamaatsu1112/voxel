use crate::builders::configs::image_configs::{
    DynamicGBufferColorImage, DynamicGBufferColorImageView, DynamicGBufferDepthImage,
    DynamicGBufferDepthImageView, DynamicGBufferImage, DynamicGBufferImageView, GBufferColorImage,
    GBufferColorImageView, GBufferDepthImage, GBufferDepthImageView, GBufferImage,
    GBufferImageView,
};
use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;

use ash::vk;
use std::marker::PhantomData;
use std::ptr;

/// Begin rendering to G-Buffer
#[derive(Clone)]
pub struct BeginGBufferRendering {}

impl VkGraphicsRecordable for BeginGBufferRendering {
    fn record(context: &mut VkGraphicsRecordContext) {
        let (width, height) = context.get_swapchain_extent();
        let extent = vk::Extent2D { width, height };
        let gbuffer_image = context
            .get_per_frame_image::<GBufferImage>(0)
            .expect("Failed to get G-Buffer image");
        let gbuffer_image_view = context
            .get_per_frame_image_view::<GBufferImageView>(0)
            .expect("Failed to get G-Buffer image view");
        let gbuffer_depth_image = context
            .get_per_frame_image::<GBufferDepthImage>(0)
            .expect("Failed to get G-Buffer depth image");
        let gbuffer_depth_image_view = context
            .get_per_frame_image_view::<GBufferDepthImageView>(0)
            .expect("Failed to get G-Buffer depth image view");

        let clear_uint = [0, 0, 0, 0];
        let clear_value_gbuffer = vk::ClearValue {
            color: vk::ClearColorValue { uint32: clear_uint },
        };

        let clear_float = [0.0, 0.0, 0.0, 0.0];
        let clear_value_depth = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear_float,
            },
        };

        let gbuffer_attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: ptr::null(),
            image_view: gbuffer_image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            clear_value: clear_value_gbuffer,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            _marker: PhantomData,
        };

        let gbuffer_depth_attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: ptr::null(),
            image_view: gbuffer_depth_image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            clear_value: clear_value_depth,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            _marker: PhantomData,
        };

        let attachments = [gbuffer_attachment_info, gbuffer_depth_attachment_info];

        let barrier_voxel = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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

        let barrier_depth = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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

        let barriers = [barrier_voxel, barrier_depth];

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
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
        context.cmd_begin_rendering(&rendering_info);
    }
}

/// Begin rendering to color G-Buffer (for color bit accumulation)
#[derive(Clone)]
pub struct BeginColorGBufferRendering {}

impl VkGraphicsRecordable for BeginColorGBufferRendering {
    fn record(context: &mut VkGraphicsRecordContext) {
        let (width, height) = context.get_swapchain_extent();
        let extent = vk::Extent2D { width, height };
        let gbuffer_color_image = context
            .get_per_frame_image::<GBufferColorImage>(0)
            .expect("Failed to get color G-Buffer image");
        let gbuffer_color_image_view = context
            .get_per_frame_image_view::<GBufferColorImageView>(0)
            .expect("Failed to get color G-Buffer image view");

        let clear_color = [0x00, 0x00, 0x00, 0xFF];
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                uint32: clear_color,
            },
        };

        let attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: ptr::null(),
            image_view: gbuffer_color_image_view,
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
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
        context.cmd_begin_rendering(&rendering_info);
    }
}

/// Begin rendering to Dynamic G-Buffer
#[derive(Clone)]
pub struct BeginDynamicGBufferRendering {}

impl VkGraphicsRecordable for BeginDynamicGBufferRendering {
    fn record(context: &mut VkGraphicsRecordContext) {
        let (width, height) = context.get_swapchain_extent();
        let extent = vk::Extent2D { width, height };
        let gbuffer_image = context
            .get_per_frame_image::<DynamicGBufferImage>(0)
            .expect("Failed to get Dynamic G-Buffer image");
        let gbuffer_image_view = context
            .get_per_frame_image_view::<DynamicGBufferImageView>(0)
            .expect("Failed to get Dynamic G-Buffer image view");
        let gbuffer_depth_image = context
            .get_per_frame_image::<DynamicGBufferDepthImage>(0)
            .expect("Failed to get Dynamic G-Buffer depth image");
        let gbuffer_depth_image_view = context
            .get_per_frame_image_view::<DynamicGBufferDepthImageView>(0)
            .expect("Failed to get Dynamic G-Buffer depth image view");

        let clear_uint = [0, 0, 0, 0];
        let clear_value_gbuffer = vk::ClearValue {
            color: vk::ClearColorValue { uint32: clear_uint },
        };

        let clear_float = [0.0, 0.0, 0.0, 0.0];
        let clear_value_depth = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear_float,
            },
        };

        let gbuffer_attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: ptr::null(),
            image_view: gbuffer_image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            clear_value: clear_value_gbuffer,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            _marker: PhantomData,
        };

        let gbuffer_depth_attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: ptr::null(),
            image_view: gbuffer_depth_image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            clear_value: clear_value_depth,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            _marker: PhantomData,
        };

        let attachments = [gbuffer_attachment_info, gbuffer_depth_attachment_info];

        let barrier_voxel = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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

        let barrier_depth = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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

        let barriers = [barrier_voxel, barrier_depth];

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
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
        context.cmd_begin_rendering(&rendering_info);
    }
}

/// Begin rendering to Dynamic color G-Buffer (for color bit accumulation)
#[derive(Clone)]
pub struct BeginDynamicColorGBufferRendering {}

impl VkGraphicsRecordable for BeginDynamicColorGBufferRendering {
    fn record(context: &mut VkGraphicsRecordContext) {
        let (width, height) = context.get_swapchain_extent();
        let extent = vk::Extent2D { width, height };
        let gbuffer_color_image = context
            .get_per_frame_image::<DynamicGBufferColorImage>(0)
            .expect("Failed to get Dynamic color G-Buffer image");
        let gbuffer_color_image_view = context
            .get_per_frame_image_view::<DynamicGBufferColorImageView>(0)
            .expect("Failed to get Dynamic color G-Buffer image view");

        let clear_color = [0x00, 0x00, 0x00, 0xFF];
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                uint32: clear_color,
            },
        };

        let attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: ptr::null(),
            image_view: gbuffer_color_image_view,
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
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
        context.cmd_begin_rendering(&rendering_info);
    }
}
