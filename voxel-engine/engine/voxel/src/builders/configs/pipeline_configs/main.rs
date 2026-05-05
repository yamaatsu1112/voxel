use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::vertex::Vertex;
use crate::vulkan::pipeline_config::{ColorAttachmentFormat, GraphicsPipelineConfig};
use crate::vulkan::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT;

pub const MAIN_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 9] = [
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
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 4,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
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
    // Dynamic G-Buffer bindings (6, 7, 8)
    vk::DescriptorSetLayoutBinding {
        binding: 6,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 7,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 8,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
];

pub const MAIN_STATIC_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 1] =
    [vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    }];

pub const MAIN_POOL_SIZES: &[vk::DescriptorPoolSize] = &[
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: (MAX_FRAMES_IN_FLIGHT * 2) as u32,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        // 6 images per frame: static (voxel + depth + color) + dynamic (voxel + depth + color)
        descriptor_count: (MAX_FRAMES_IN_FLIGHT * 6) as u32,
    },
];

pub const MAIN_SET_COUNT: usize = MAX_FRAMES_IN_FLIGHT;
pub const MAIN_STATIC_POOL_SIZES: &[vk::DescriptorPoolSize] = &[vk::DescriptorPoolSize {
    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    descriptor_count: 1,
}];
pub const MAIN_STATIC_SET_COUNT: usize = 1;

pub const MAIN_VERT_PATH: &str = concat!(env!("VOXEL_SHADER_DIR"), "/vert.spv");
pub const MAIN_FRAG_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/frag_3dtex_",
    env!("VOXEL_ACTIVE_SHADER_SUFFIX"),
    ".spv"
);

pub const MAIN_BLEND_ATTACHMENTS: [vk::PipelineColorBlendAttachmentState; 1] =
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

// Vertex binding description for main pipeline
pub const MAIN_BINDING_DESCRIPTIONS: [vk::VertexInputBindingDescription; 1] =
    [vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];

// Vertex attribute descriptions for main pipeline
pub const MAIN_ATTRIBUTE_DESCRIPTIONS: [vk::VertexInputAttributeDescription; 2] = [
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

engine_macro::define_graphics_pipeline! {
    /// Marker for main graphics pipeline
    pub struct MainPipeline;
    config = GraphicsPipelineConfig {
        vert_shader_path: MAIN_VERT_PATH,
        frag_shader_path: MAIN_FRAG_PATH,
        descriptor_set_layouts: &[
            &MAIN_DESCRIPTOR_SET_LAYOUT_BINDINGS,
            &MAIN_STATIC_DESCRIPTOR_SET_LAYOUT_BINDINGS,
        ],
        push_constant_ranges: &[],
        color_attachment_formats: &[ColorAttachmentFormat::Swapchain],
        binding_descriptions: &MAIN_BINDING_DESCRIPTIONS,
        attribute_descriptions: &MAIN_ATTRIBUTE_DESCRIPTIONS,
        blend_attachments: &MAIN_BLEND_ATTACHMENTS,
        logic_op_enable: false,
        logic_op: vk::LogicOp::COPY,
        blend_constants: [0.0, 0.0, 0.0, 0.0],
    };
}
