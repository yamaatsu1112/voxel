use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::vertex::Vertex;
use crate::vulkan::pipeline_config::{ColorAttachmentFormat, GraphicsPipelineConfig};
use crate::vulkan::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT;

// Descriptor set layout for color write pipeline
// Needs: UBO (binding 0), SVO nodes (binding 1), SVO leaves (binding 2),
// ID SVO node array (binding 3), ID SVO leaf array (binding 4), G-Buffer voxel data (binding 5)
pub const GBUFFER_COLOR_WRITE_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 6] = [
    vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 1,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 2,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 3,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 8, // 8 ID SVO buffers (descriptor array)
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 4,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 8, // 8 ID SVO leaf buffers (descriptor array)
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 5,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
];

pub const GBUFFER_COLOR_WRITE_POOL_SIZES: &[vk::DescriptorPoolSize] = &[
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: (MAX_FRAMES_IN_FLIGHT * 2 + MAX_FRAMES_IN_FLIGHT * 16) as u32,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: MAX_FRAMES_IN_FLIGHT as u32, // G-Buffer voxel data
    },
];

pub const GBUFFER_COLOR_WRITE_SET_COUNT: usize = MAX_FRAMES_IN_FLIGHT;

pub const GBUFFER_COLOR_WRITE_VERT_PATH: &str = concat!(env!("VOXEL_SHADER_DIR"), "/vert.spv"); // Reuse main vertex shader
pub const GBUFFER_COLOR_WRITE_FRAG_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/gbuffer_color_write_frag_",
    env!("VOXEL_ACTIVE_SHADER_SUFFIX"),
    ".spv"
);
pub const GBUFFER_ID_WRITE_FRAG_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/gbuffer_id_write_frag_",
    env!("VOXEL_ACTIVE_SHADER_SUFFIX"),
    ".spv"
);

// Vertex binding description for G-Buffer color write pipeline (same as main)
pub const GBUFFER_COLOR_WRITE_BINDING_DESCRIPTIONS: [vk::VertexInputBindingDescription; 1] =
    [vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];

// Vertex attribute descriptions for G-Buffer color write pipeline (same as main)
pub const GBUFFER_COLOR_WRITE_ATTRIBUTE_DESCRIPTIONS: [vk::VertexInputAttributeDescription; 2] = [
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 0,
        format: vk::Format::R32G32_SFLOAT,
        offset: std::mem::offset_of!(Vertex, position) as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 1,
        format: vk::Format::R32G32_SFLOAT,
        offset: std::mem::offset_of!(Vertex, tex_coord) as u32,
    },
];

// Push constants for passing bit index to shader
#[repr(C)]
pub struct ColorBitPushConstants {
    pub bit_index: u32,
}

pub const GBUFFER_COLOR_WRITE_PUSH_CONSTANT_RANGES: [vk::PushConstantRange; 1] =
    [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: std::mem::size_of::<ColorBitPushConstants>() as u32,
    }];

pub const GBUFFER_COLOR_WRITE_BLEND_ATTACHMENTS: [vk::PipelineColorBlendAttachmentState; 1] =
    [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE, // Must be FALSE when using logic ops
        src_color_blend_factor: vk::BlendFactor::ZERO,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ZERO,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    }];

engine_macro::define_graphics_pipeline! {
    /// Marker for G-Buffer color write pipeline
    pub struct GBufferColorWritePipeline;
    config = GraphicsPipelineConfig {
        vert_shader_path: GBUFFER_COLOR_WRITE_VERT_PATH,
        frag_shader_path: GBUFFER_ID_WRITE_FRAG_PATH,
        descriptor_set_layouts: &[&GBUFFER_COLOR_WRITE_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &GBUFFER_COLOR_WRITE_PUSH_CONSTANT_RANGES,
        color_attachment_formats: &[ColorAttachmentFormat::R8G8B8A8Uint], // Color G-Buffer format
        binding_descriptions: &GBUFFER_COLOR_WRITE_BINDING_DESCRIPTIONS,
        attribute_descriptions: &GBUFFER_COLOR_WRITE_ATTRIBUTE_DESCRIPTIONS,
        blend_attachments: &GBUFFER_COLOR_WRITE_BLEND_ATTACHMENTS,
        logic_op_enable: true,
        logic_op: vk::LogicOp::OR,
        blend_constants: [0.0, 0.0, 0.0, 0.0],
    };
}
