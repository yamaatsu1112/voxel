use crate::rendering::GpuMutexList;
use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::{VkComputeRecordable, VkGraphicsRecordable};
use std::any::TypeId;

pub trait ComputeSubmitGroup: 'static {
    type Mutex: GpuMutexList;
    type Recordables: VkComputeRecordable + Default + Send + 'static;
}

pub trait GraphicsSubmitGroup: Send + Sync + 'static {
    type Mutex: GpuMutexList;
    type Recordables: VkGraphicsRecordable;
}

pub(crate) trait ErasedGraphicsSubmitGroup: Send + Sync {
    fn mutex_ids(&self) -> Vec<TypeId>;
    fn record(&self, context: &mut VkGraphicsRecordContext);
}

impl<T: GraphicsSubmitGroup> ErasedGraphicsSubmitGroup for T {
    fn mutex_ids(&self) -> Vec<TypeId> {
        <T::Mutex as GpuMutexList>::mutex_ids()
    }

    fn record(&self, context: &mut VkGraphicsRecordContext) {
        <T::Recordables as VkGraphicsRecordable>::record(context);
    }
}
