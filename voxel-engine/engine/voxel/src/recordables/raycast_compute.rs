use crate::voxel::svo::svo_config::{RaycastComputeDescriptorSet, RaycastComputePipeline};
use crate::vulkan::record_context::VkComputeRecordContext;
use crate::vulkan::recordable::VkComputeRecordable;

#[derive(Clone, Default)]
pub struct RaycastComputePipelineRecordable;

impl RaycastComputePipelineRecordable {
    pub fn new() -> Self {
        Self
    }

    pub fn record_compute(context: &mut VkComputeRecordContext) {
        context.cmd_bind_compute_pipeline::<RaycastComputePipeline>();
        context
            .cmd_bind_descriptor_set::<RaycastComputePipeline, RaycastComputeDescriptorSet>(0, &[]);
        context.cmd_dispatch(1, 1, 1);
    }
}

impl VkComputeRecordable for RaycastComputePipelineRecordable {
    fn record(context: &mut VkComputeRecordContext) {
        Self::record_compute(context);
    }
}
