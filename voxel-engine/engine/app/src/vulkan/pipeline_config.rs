use ash::vk;

/// Color attachment format specification for pipeline creation.
/// Abstracts common formats to simplify pipeline configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorAttachmentFormat {
    /// Use swapchain format (determined at runtime)
    Swapchain,
    /// R32G32B32A32_UINT format (e.g., G-Buffer voxel data)
    R32G32B32A32Uint,
    /// R32_SFLOAT format (e.g., G-Buffer depth)
    R32Sfloat,
    /// R8G8B8A8_UINT format (e.g., G-Buffer color)
    R8G8B8A8Uint,
}

impl ColorAttachmentFormat {
    /// Resolve the format to a concrete vk::Format
    pub fn resolve(&self, swapchain_format: vk::Format) -> vk::Format {
        match self {
            ColorAttachmentFormat::Swapchain => swapchain_format,
            ColorAttachmentFormat::R32G32B32A32Uint => vk::Format::R32G32B32A32_UINT,
            ColorAttachmentFormat::R32Sfloat => vk::Format::R32_SFLOAT,
            ColorAttachmentFormat::R8G8B8A8Uint => vk::Format::R8G8B8A8_UINT,
        }
    }
}

/// Configuration for creating a graphics pipeline.
/// Contains all parameters needed to create a complete graphics pipeline.
pub struct GraphicsPipelineConfig {
    /// Absolute path to vertex shader SPIR-V file
    pub vert_shader_path: &'static str,
    /// Absolute path to fragment shader SPIR-V file
    pub frag_shader_path: &'static str,
    /// Descriptor set layout bindings for each set
    pub descriptor_set_layouts: &'static [&'static [vk::DescriptorSetLayoutBinding<'static>]],
    /// Push constant ranges
    pub push_constant_ranges: &'static [vk::PushConstantRange],
    /// Color attachment formats for the pipeline
    pub color_attachment_formats: &'static [ColorAttachmentFormat],
    /// Vertex input binding descriptions
    pub binding_descriptions: &'static [vk::VertexInputBindingDescription],
    /// Vertex input attribute descriptions
    pub attribute_descriptions: &'static [vk::VertexInputAttributeDescription],
    /// Color blend attachment states
    pub blend_attachments: &'static [vk::PipelineColorBlendAttachmentState],
    /// Enable logic operations (for color bit accumulation)
    pub logic_op_enable: bool,
    /// Logic operation to use when logic_op_enable is true
    pub logic_op: vk::LogicOp,
    /// Blend constants
    pub blend_constants: [f32; 4],
}

/// Configuration for creating a compute pipeline.
/// Contains all parameters needed to create a complete compute pipeline.
pub struct ComputePipelineConfig {
    /// Absolute path to compute shader SPIR-V file
    pub shader_path: &'static str,
    /// Descriptor set layout bindings for each set
    pub descriptor_set_layouts: &'static [&'static [vk::DescriptorSetLayoutBinding<'static>]],
    /// Push constant ranges
    pub push_constant_ranges: &'static [vk::PushConstantRange],
}
