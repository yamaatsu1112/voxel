use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::components::ui::UIInstance;
use crate::vulkan::pipeline_config::{ColorAttachmentFormat, GraphicsPipelineConfig};
use crate::vulkan::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT;

// Vertex binding description for UI pipeline
pub const UI_BINDING_DESCRIPTIONS: [vk::VertexInputBindingDescription; 1] =
    [vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<UIInstance>() as u32,
        input_rate: vk::VertexInputRate::INSTANCE,
    }];

// Vertex attribute descriptions for UI pipeline
pub const UI_ATTRIBUTE_DESCRIPTIONS: [vk::VertexInputAttributeDescription; 8] = [
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 0,
        format: vk::Format::R32_SFLOAT,
        offset: std::mem::offset_of!(UIInstance, position_x) as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 1,
        format: vk::Format::R32_SFLOAT,
        offset: std::mem::offset_of!(UIInstance, position_y) as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 2,
        format: vk::Format::R32_SFLOAT,
        offset: std::mem::offset_of!(UIInstance, width) as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 3,
        format: vk::Format::R32_SFLOAT,
        offset: std::mem::offset_of!(UIInstance, height) as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 4,
        format: vk::Format::R32_SFLOAT,
        offset: std::mem::offset_of!(UIInstance, atlas_u_offset) as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 5,
        format: vk::Format::R32_SFLOAT,
        offset: std::mem::offset_of!(UIInstance, atlas_v_offset) as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 6,
        format: vk::Format::R32_SFLOAT,
        offset: std::mem::offset_of!(UIInstance, atlas_u_size) as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 7,
        format: vk::Format::R32_SFLOAT,
        offset: std::mem::offset_of!(UIInstance, atlas_v_size) as u32,
    },
];

pub const UI_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 1] =
    [vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    }];

pub const UI_POOL_SIZES: &[vk::DescriptorPoolSize] = &[vk::DescriptorPoolSize {
    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    descriptor_count: UI_SET_COUNT as u32,
}];

pub const UI_SET_COUNT: usize = MAX_FRAMES_IN_FLIGHT;

pub const UI_VERT_PATH: &str = concat!(env!("UI_SHADER_DIR"), "/ui-vert.spv");
pub const UI_FRAG_PATH: &str = concat!(env!("UI_SHADER_DIR"), "/ui-frag.spv");

pub const UI_INSTANCE_COUNT: usize = 32;

pub const UI_BLEND_ATTACHMENTS: [vk::PipelineColorBlendAttachmentState; 1] =
    [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    }];

engine_macro::define_graphics_pipeline! {
    /// Marker for UI pipeline
    pub struct UIPipeline;
    config = GraphicsPipelineConfig {
        vert_shader_path: UI_VERT_PATH,
        frag_shader_path: UI_FRAG_PATH,
        descriptor_set_layouts: &[&UI_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &[],
        color_attachment_formats: &[ColorAttachmentFormat::Swapchain],
        binding_descriptions: &UI_BINDING_DESCRIPTIONS,
        attribute_descriptions: &UI_ATTRIBUTE_DESCRIPTIONS,
        blend_attachments: &UI_BLEND_ATTACHMENTS,
        logic_op_enable: false,
        logic_op: vk::LogicOp::COPY,
        blend_constants: [0.0, 0.0, 0.0, 0.0],
    };
}
