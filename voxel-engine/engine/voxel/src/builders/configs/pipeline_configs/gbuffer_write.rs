use ash::vk;

use crate::builders::configs::pipeline_configs::main::MAIN_DESCRIPTOR_SET_LAYOUT_BINDINGS;
use crate::vertex::Vertex;
use crate::vulkan::pipeline_config::{ColorAttachmentFormat, GraphicsPipelineConfig};

// NOTE: GBufferWritePipeline uses MainDescriptorSet (shared with main pipeline)
// so there's no dedicated descriptor set count for it.

pub const GBUFFER_WRITE_VERT_PATH: &str = concat!(env!("VOXEL_SHADER_DIR"), "/vert.spv"); // Reuse main vertex shader
pub const GBUFFER_WRITE_FRAG_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/gbuffer_write_frag_",
    env!("VOXEL_ACTIVE_SHADER_SUFFIX"),
    ".spv"
);

// Vertex binding description for G-Buffer write pipeline (same as main)
pub const GBUFFER_WRITE_BINDING_DESCRIPTIONS: [vk::VertexInputBindingDescription; 1] =
    [vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];

// Vertex attribute descriptions for G-Buffer write pipeline (same as main)
pub const GBUFFER_WRITE_ATTRIBUTE_DESCRIPTIONS: [vk::VertexInputAttributeDescription; 2] = [
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

pub const GBUFFER_WRITE_BLEND_ATTACHMENTS: [vk::PipelineColorBlendAttachmentState; 2] = [
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
    /// Marker for G-Buffer write pipeline
    pub struct GBufferWritePipeline;
    config = GraphicsPipelineConfig {
        vert_shader_path: GBUFFER_WRITE_VERT_PATH,
        frag_shader_path: GBUFFER_WRITE_FRAG_PATH,
        descriptor_set_layouts: &[&MAIN_DESCRIPTOR_SET_LAYOUT_BINDINGS], // Reuses main descriptor set layout
        push_constant_ranges: &[],
        color_attachment_formats: &[
            ColorAttachmentFormat::R32G32B32A32Uint, // voxel data
            ColorAttachmentFormat::R32Sfloat,        // depth
        ],
        binding_descriptions: &GBUFFER_WRITE_BINDING_DESCRIPTIONS,
        attribute_descriptions: &GBUFFER_WRITE_ATTRIBUTE_DESCRIPTIONS,
        blend_attachments: &GBUFFER_WRITE_BLEND_ATTACHMENTS,
        logic_op_enable: false,
        logic_op: vk::LogicOp::COPY,
        blend_constants: [0.0, 0.0, 0.0, 0.0],
    };
}
