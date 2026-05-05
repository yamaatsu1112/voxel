#[allow(dead_code)]
pub(crate) mod buffers;
mod cpu;
#[allow(dead_code)]
pub(crate) mod descriptor_sets;
#[allow(dead_code)]
pub(crate) mod pipelines;
#[allow(dead_code)]
pub(crate) mod recordables;
#[allow(dead_code)]
pub(crate) mod register;

pub use cpu::SVOGrouped;

use super::svo_variant::SvoVariant;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

#[allow(dead_code)]
pub struct SvoGroupedVariant;
#[allow(dead_code)]
pub(crate) type Variant = SvoGroupedVariant;

impl SvoVariant for SvoGroupedVariant {
    type CpuSvo = SVOGrouped;

    type VoxelRecordable = recordables::VoxelPipelineRecordable;
    type VoxelDestroyRecordable = recordables::VoxelDestroyPipelineRecordable;
    type DynamicVoxelRecordable = recordables::DynamicVoxelPipelineRecordable;

    type CopySVO = recordables::CopySVO;
    type CopyDynamicSVO = recordables::CopyDynamicSVO;

    type RaycastComputeDescriptorSet = descriptor_sets::RaycastComputeDescriptorSet;
    type RaycastComputePipeline = pipelines::RaycastComputePipeline;
    type CollisionComputeDescriptorSet = descriptor_sets::CollisionComputeDescriptorSet;
    type CollisionComputePipeline = pipelines::CollisionComputePipeline;

    fn register_resources(renderer: &mut Renderer<VulkanRenderer>) {
        register::register_resources(renderer);
    }
}
