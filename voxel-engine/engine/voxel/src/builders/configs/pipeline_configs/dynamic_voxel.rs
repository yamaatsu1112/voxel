use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;
use crate::builders::configs::pipeline_configs::voxel::VoxelPushConstants;

// LAYER_TYPE_CLEAR_SVO - used to clear all dynamic SVO data each frame
pub const LAYER_TYPE_CLEAR_SVO: u32 = 10;

// Descriptor set layout (Set 0): split SVO storage buffers + temp buffers + ComputeUniformBuffer
pub const DYNAMIC_VOXEL_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 13] = [
    // Binding 0: Dynamic SVO node buffers
    vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 1: Dynamic SVO leaf buffers
    vk::DescriptorSetLayoutBinding {
        binding: 1,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 2: Dynamic SVO counter buffers
    vk::DescriptorSetLayoutBinding {
        binding: 2,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 3: Dynamic SVO free-node index buffers
    vk::DescriptorSetLayoutBinding {
        binding: 3,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 4: Dynamic SVO free-leaf index buffers
    vk::DescriptorSetLayoutBinding {
        binding: 4,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 5: Dynamic leaf edit command buffers
    vk::DescriptorSetLayoutBinding {
        binding: 5,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Bindings 6-10: Temporary buffers
    vk::DescriptorSetLayoutBinding {
        binding: 6,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 7,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 8,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 9,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 10,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 11,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 12,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
];

pub const DYNAMIC_VOXEL_SET_COUNT: usize = 1;

pub const DYNAMIC_VOXEL_POOL_SIZES: &[vk::DescriptorPoolSize] = &[
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: (NUM_SVO_DATA as u32) * 11 + 1,
    },
];

pub const DYNAMIC_VOXEL_PUSH_CONSTANT_RANGES: [vk::PushConstantRange; 1] =
    [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: std::mem::size_of::<VoxelPushConstants>() as u32,
    }];
