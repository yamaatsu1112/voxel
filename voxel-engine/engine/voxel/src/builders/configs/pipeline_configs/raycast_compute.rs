use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::vulkan::pipeline_config::ComputePipelineConfig;

// Descriptor set layout for raycast compute pipeline
// Binding 0: Uniform Buffer (UBO)
// Binding 1: Storage Buffer (SVO nodes - read only)
// Binding 2: Storage Buffer (SVO leaves - read only)
// Binding 3: Storage Buffer (Raycast Result Buffer - write)
pub const RAYCAST_COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 4] = [
    vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 1,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 2,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 3,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
];

pub const RAYCAST_COMPUTE_POOL_SIZES: &[vk::DescriptorPoolSize] = &[
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3, // nodes + leaves + result buffer
    },
];

pub const RAYCAST_COMPUTE_SET_COUNT: usize = 1;

pub const RAYCAST_COMPUTE_SHADER_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/raycast_compute_",
    env!("VOXEL_ACTIVE_SHADER_SUFFIX"),
    ".spv"
);

engine_macro::define_compute_pipeline! {
    /// Marker for raycast compute pipeline
    pub struct RaycastComputePipeline;
    config = ComputePipelineConfig {
        shader_path: RAYCAST_COMPUTE_SHADER_PATH,
        descriptor_set_layouts: &[&RAYCAST_COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &[],
    };
}
