use ash::vk;

use crate::builders::configs::buffer_configs::compute::NUM_SVO_DATA;

pub const DESTROY_COMPUTE_POOL_SIZES: &[vk::DescriptorPoolSize] = &[
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
    },
    vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: (NUM_SVO_DATA as u32) * 11 + 1,
    },
];

pub const DESTROY_COMPUTE_SET_COUNT: usize = 1;
