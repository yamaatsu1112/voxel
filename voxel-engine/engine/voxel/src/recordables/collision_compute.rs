use crate::voxel::svo::svo_config::{CollisionComputeDescriptorSet, CollisionComputePipeline};
use crate::vulkan::record_context::VkComputeRecordContext;
use crate::vulkan::recordable::VkComputeRecordable;

#[derive(Clone, Default)]
pub struct CollisionComputePipelineRecordable;

impl CollisionComputePipelineRecordable {
    pub fn new() -> Self {
        Self
    }

    pub fn record_compute(context: &mut VkComputeRecordContext) {
        context.cmd_bind_compute_pipeline::<CollisionComputePipeline>();
        context.cmd_bind_descriptor_set::<CollisionComputePipeline, CollisionComputeDescriptorSet>(
            0,
            &[],
        );
        context.cmd_dispatch(1, 1, 1);
    }
}

impl VkComputeRecordable for CollisionComputePipelineRecordable {
    fn record(context: &mut VkComputeRecordContext) {
        Self::record_compute(context);
    }
}
