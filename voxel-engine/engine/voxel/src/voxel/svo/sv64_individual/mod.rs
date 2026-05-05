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

pub use cpu::SV64Individual;

use super::svo_variant::SvoVariant;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

#[allow(dead_code)]
pub struct Sv64IndividualVariant;
#[allow(dead_code)]
pub(crate) type Variant = Sv64IndividualVariant;

impl SvoVariant for Sv64IndividualVariant {
    type CpuSvo = SV64Individual;

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
