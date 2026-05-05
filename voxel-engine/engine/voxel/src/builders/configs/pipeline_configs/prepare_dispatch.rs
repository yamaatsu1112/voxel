use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::vulkan::pipeline_config::ComputePipelineConfig;

pub const PREPARE_DISPATCH_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 2] = [
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
];

pub const PREPARE_DISPATCH_POOL_SIZES: &[vk::DescriptorPoolSize] = &[vk::DescriptorPoolSize {
    ty: vk::DescriptorType::STORAGE_BUFFER,
    descriptor_count: 2,
}];

pub const PREPARE_DISPATCH_SET_COUNT: usize = 1;

pub const PREPARE_DISPATCH_SHADER_PATH: &str =
    concat!(env!("VOXEL_SHADER_DIR"), "/prepare_dispatch.spv");

engine_macro::define_compute_pipeline! {
    pub struct PrepareDispatchPipeline;
    config = ComputePipelineConfig {
        shader_path: PREPARE_DISPATCH_SHADER_PATH,
        descriptor_set_layouts: &[&PREPARE_DISPATCH_DESCRIPTOR_SET_LAYOUT_BINDINGS],
        push_constant_ranges: &[],
    };
}
