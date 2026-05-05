use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::vertex::Vertex;
use crate::vulkan::pipeline_config::{ColorAttachmentFormat, GraphicsPipelineConfig};
use crate::vulkan::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT;

// Descriptor set layout for Dynamic G-Buffer write pipeline
// Needs: UBO (binding 0), Dynamic SVO nodes (binding 1), Dynamic SVO leaves (binding 2)
pub const DYNAMIC_GBUFFER_WRITE_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding;
    3] = [
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
];

pub const DYNAMIC_GBUFFER_WRITE_POOL_SIZES: &[vk::DescriptorPoolSize] = &[
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: (MAX_FRAMES_IN_FLIGHT * 2) as u32,
    },
];

pub const DYNAMIC_GBUFFER_WRITE_SET_COUNT: usize = MAX_FRAMES_IN_FLIGHT;

pub const DYNAMIC_GBUFFER_WRITE_VERT_PATH: &str = concat!(env!("VOXEL_SHADER_DIR"), "/vert.spv"); // Reuse main vertex shader
pub const DYNAMIC_GBUFFER_WRITE_FRAG_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/dynamic_gbuffer_write_frag_",
    env!("VOXEL_ACTIVE_SHADER_SUFFIX"),
    ".spv"
);

// Vertex binding description for Dynamic G-Buffer write pipeline (same as main)
pub const DYNAMIC_GBUFFER_WRITE_BINDING_DESCRIPTIONS: [vk::VertexInputBindingDescription; 1] =
    [vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];

// Vertex attribute descriptions for Dynamic G-Buffer write pipeline (same as main)
pub const DYNAMIC_GBUFFER_WRITE_ATTRIBUTE_DESCRIPTIONS: [vk::VertexInputAttributeDescription; 2] = [
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

pub const DYNAMIC_GBUFFER_WRITE_BLEND_ATTACHMENTS: [vk::PipelineColorBlendAttachmentState; 2] = [
    vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    },
    vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    },
];

engine_macro::define_graphics_pipeline! {
    /// Marker for dynamic G-Buffer write pipeline
    pub struct DynamicGBufferWritePipeline;
    config = GraphicsPipelineConfig {
        vert_shader_path: DYNAMIC_GBUFFER_WRITE_VERT_PATH,
        frag_shader_path: DYNAMIC_GBUFFER_WRITE_FRAG_PATH,
        descriptor_set_layouts: &[&DYNAMIC_GBUFFER_WRITE_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &[],
        color_attachment_formats: &[
            ColorAttachmentFormat::R32G32B32A32Uint, // voxel data
            ColorAttachmentFormat::R32Sfloat,        // depth
        ],
        binding_descriptions: &DYNAMIC_GBUFFER_WRITE_BINDING_DESCRIPTIONS,
        attribute_descriptions: &DYNAMIC_GBUFFER_WRITE_ATTRIBUTE_DESCRIPTIONS,
        blend_attachments: &DYNAMIC_GBUFFER_WRITE_BLEND_ATTACHMENTS,
        logic_op_enable: false,
        logic_op: vk::LogicOp::COPY,
        blend_constants: [0.0, 0.0, 0.0, 0.0],
    };
}
