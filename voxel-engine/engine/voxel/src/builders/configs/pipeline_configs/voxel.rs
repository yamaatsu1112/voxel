use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;

// Push constant structure for voxel compute shader dispatch
#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct VoxelPushConstants {
    pub depth: u32,
    pub layer_type: u32, // Pipeline phase selector
    pub phase: u32,      // Phase within parallel edit pipeline
    pub svo_index: u32, // Index of the SVOData being processed (0 = occupancy, 1-24 = color bit planes)
}

// Descriptor set layout (Set 0): split SVO storage buffers + temp buffers + ComputeUniformBuffer
pub const COMPUTE_DESCRIPTOR_SET_LAYOUT_BINDINGS: [vk::DescriptorSetLayoutBinding; 13] = [
    // Binding 0: SVO node buffers
    vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 1: SVO leaf buffers
    vk::DescriptorSetLayoutBinding {
        binding: 1,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 2: SVO counter buffers
    vk::DescriptorSetLayoutBinding {
        binding: 2,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 3: SVO free-node index buffers
    vk::DescriptorSetLayoutBinding {
        binding: 3,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 4: SVO free-leaf index buffers
    vk::DescriptorSetLayoutBinding {
        binding: 4,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 5: leaf edit command buffers
    vk::DescriptorSetLayoutBinding {
        binding: 5,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 6: temp buffers
    vk::DescriptorSetLayoutBinding {
        binding: 6,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 7: temp buffers
    vk::DescriptorSetLayoutBinding {
        binding: 7,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 8: temp buffers
    vk::DescriptorSetLayoutBinding {
        binding: 8,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 9: temp buffers
    vk::DescriptorSetLayoutBinding {
        binding: 9,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 10: temp buffers
    vk::DescriptorSetLayoutBinding {
        binding: 10,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: NUM_SVO_DATA as u32,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 11: Compute uniform buffer
    vk::DescriptorSetLayoutBinding {
        binding: 11,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
    // Binding 12: Command count buffer
    vk::DescriptorSetLayoutBinding {
        binding: 12,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: ptr::null(),
        _marker: PhantomData,
    },
];

pub const COMPUTE_POOL_SIZES: &[vk::DescriptorPoolSize] = &[
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: (NUM_SVO_DATA as u32) * 11 + 1,
    },
];

pub const VOXEL_PUSH_CONSTANT_RANGES: [vk::PushConstantRange; 1] = [vk::PushConstantRange {
    stage_flags: vk::ShaderStageFlags::COMPUTE,
    offset: 0,
    size: std::mem::size_of::<VoxelPushConstants>() as u32,
}];

// Constants for layer types - Allocation
pub const LAYER_TYPE_COLLECT_REQUESTS: u32 = 0;
pub const LAYER_TYPE_UNIQUE_FLAGS: u32 = 3;
pub const LAYER_TYPE_PREFIX_SUM: u32 = 4;
pub const LAYER_TYPE_COMPACT: u32 = 5;
pub const LAYER_TYPE_ALLOCATE_BATCH: u32 = 6;
pub const LAYER_TYPE_UPDATE_LEAVES: u32 = 7;

// Constants for layer types - Freeing/Integration
pub const LAYER_TYPE_COLLECT_FREE_REQUESTS: u32 = 8;
pub const LAYER_TYPE_FREE_BATCH: u32 = 9;

pub const COMPUTE_SET_COUNT: usize = 1;

#[cfg(test)]
mod tests {
    use super::VoxelPushConstants;
    use crate::voxel::command::LeafEditCommand;

    #[test]
    fn gpu_buffer_layouts_match_expected_sizes() {
        assert_eq!(std::mem::size_of::<LeafEditCommand>(), 20);
        assert_eq!(std::mem::size_of::<VoxelPushConstants>(), 16);
    }
}
