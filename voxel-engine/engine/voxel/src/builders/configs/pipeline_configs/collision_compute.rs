use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::vulkan::pipeline_config::ComputePipelineConfig;

pub const COLLISION_COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 4] = [
    vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
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

pub const COLLISION_COMPUTE_POOL_SIZES: &[vk::DescriptorPoolSize] = &[vk::DescriptorPoolSize {
    ty: vk::DescriptorType::STORAGE_BUFFER,
    descriptor_count: 4,
}];

pub const COLLISION_COMPUTE_SET_COUNT: usize = 1;

pub const COLLISION_COMPUTE_SHADER_PATH: &str = concat!(
    env!("VOXEL_SHADER_DIR"),
    "/collision_compute_",
    env!("VOXEL_ACTIVE_SHADER_SUFFIX"),
    ".spv"
);

engine_macro::define_compute_pipeline! {
    pub struct CollisionComputePipeline;
    config = ComputePipelineConfig {
        shader_path: COLLISION_COMPUTE_SHADER_PATH,
        descriptor_set_layouts: &[&COLLISION_COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &[],
    };
}
