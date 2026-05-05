use super::SVOBackend;
use crate::rendering::renderer::Renderer;
use crate::vulkan::recordable::{VkComputeRecordable, VkGraphicsRecordable};
use crate::vulkan::resource_config::{ComputePipelineMarker, DescriptorSetMarker};
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub trait SvoVariant: 'static {
    type CpuSvo: SVOBackend + Send + Sync + 'static;

    type VoxelRecordable: VkComputeRecordable + Default + Send + 'static;
    type VoxelDestroyRecordable: VkComputeRecordable + Default + Send + 'static;
    type DynamicVoxelRecordable: VkComputeRecordable + Default + Send + 'static;

    type CopySVO: VkGraphicsRecordable + Clone + Send + 'static;
    type CopyDynamicSVO: VkGraphicsRecordable + Clone + Send + 'static;

    type RaycastComputeDescriptorSet: DescriptorSetMarker + Send + Sync + 'static;
    type RaycastComputePipeline: ComputePipelineMarker + Send + Sync + 'static;
    type CollisionComputeDescriptorSet: DescriptorSetMarker + Send + Sync + 'static;
    type CollisionComputePipeline: ComputePipelineMarker + Send + Sync + 'static;

    fn register_resources(renderer: &mut Renderer<VulkanRenderer>);
}
