use super::svo_variant::SvoVariant;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub type CurrentSvo = super::individual::Variant;

pub type VoxelPipelineRecordable = <CurrentSvo as SvoVariant>::VoxelRecordable;
pub type VoxelDestroyPipelineRecordable = <CurrentSvo as SvoVariant>::VoxelDestroyRecordable;
pub type DynamicVoxelPipelineRecordable = <CurrentSvo as SvoVariant>::DynamicVoxelRecordable;
pub type CopySVO = <CurrentSvo as SvoVariant>::CopySVO;
pub type CopyDynamicSVO = <CurrentSvo as SvoVariant>::CopyDynamicSVO;
pub type RaycastComputeDescriptorSet = <CurrentSvo as SvoVariant>::RaycastComputeDescriptorSet;
pub type RaycastComputePipeline = <CurrentSvo as SvoVariant>::RaycastComputePipeline;
pub type CollisionComputeDescriptorSet = <CurrentSvo as SvoVariant>::CollisionComputeDescriptorSet;
pub type CollisionComputePipeline = <CurrentSvo as SvoVariant>::CollisionComputePipeline;

pub(crate) fn register_resources(renderer: &mut Renderer<VulkanRenderer>) {
    <CurrentSvo as SvoVariant>::register_resources(renderer);
}
