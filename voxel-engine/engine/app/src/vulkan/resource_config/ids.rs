use super::markers::{
    BufferMarker, ComputePipelineMarker, DescriptorPoolMarker, DescriptorSetMarker,
    GraphicsPipelineMarker, ImageMarker, ImageViewMarker, ReusableCommandBufferMarker,
    SamplerMarker,
};
use crate::vulkan::resource_lifetime::PerFrame;
use std::sync::atomic::{AtomicUsize, Ordering};

/// ID for a buffer slot in the flat Vec storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub(crate) usize);

impl BufferId {
    pub const INVALID: Self = Self(usize::MAX);

    pub fn load(atomic: &AtomicUsize) -> Self {
        Self(atomic.load(Ordering::Acquire))
    }

    pub fn store(self, atomic: &AtomicUsize) {
        atomic.store(self.0, Ordering::Release);
    }
}

/// ID for an image slot in the flat Vec storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageId(pub(crate) usize);

impl ImageId {
    pub const INVALID: Self = Self(usize::MAX);

    pub fn load(atomic: &AtomicUsize) -> Self {
        Self(atomic.load(Ordering::Acquire))
    }

    pub fn store(self, atomic: &AtomicUsize) {
        atomic.store(self.0, Ordering::Release);
    }
}

/// ID for an image view slot in the flat Vec storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageViewId(pub(crate) usize);

impl ImageViewId {
    pub const INVALID: Self = Self(usize::MAX);

    pub fn load(atomic: &AtomicUsize) -> Self {
        Self(atomic.load(Ordering::Acquire))
    }

    pub fn store(self, atomic: &AtomicUsize) {
        atomic.store(self.0, Ordering::Release);
    }
}

/// ID for a sampler slot in the flat Vec storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerId(pub(crate) usize);

impl SamplerId {
    pub const INVALID: Self = Self(usize::MAX);

    pub fn load(atomic: &AtomicUsize) -> Self {
        Self(atomic.load(Ordering::Acquire))
    }

    pub fn store(self, atomic: &AtomicUsize) {
        atomic.store(self.0, Ordering::Release);
    }
}

/// ID for a descriptor set slot in the flat Vec storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorSetId(pub(crate) usize);

impl DescriptorSetId {
    pub const INVALID: Self = Self(usize::MAX);

    pub fn load(atomic: &AtomicUsize) -> Self {
        Self(atomic.load(Ordering::Acquire))
    }

    pub fn store(self, atomic: &AtomicUsize) {
        atomic.store(self.0, Ordering::Release);
    }
}

/// ID for a descriptor pool slot in the flat Vec storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorPoolId(pub(crate) usize);

impl DescriptorPoolId {
    pub const INVALID: Self = Self(usize::MAX);

    pub fn load(atomic: &AtomicUsize) -> Self {
        Self(atomic.load(Ordering::Acquire))
    }

    pub fn store(self, atomic: &AtomicUsize) {
        atomic.store(self.0, Ordering::Release);
    }
}

/// ID for a graphics pipeline slot in the flat Vec storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphicsPipelineId(pub(crate) usize);

impl GraphicsPipelineId {
    pub const INVALID: Self = Self(usize::MAX);

    pub fn load(atomic: &AtomicUsize) -> Self {
        Self(atomic.load(Ordering::Acquire))
    }

    pub fn store(self, atomic: &AtomicUsize) {
        atomic.store(self.0, Ordering::Release);
    }
}

/// ID for a compute pipeline slot in the flat Vec storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePipelineId(pub(crate) usize);

impl ComputePipelineId {
    pub const INVALID: Self = Self(usize::MAX);

    pub fn load(atomic: &AtomicUsize) -> Self {
        Self(atomic.load(Ordering::Acquire))
    }

    pub fn store(self, atomic: &AtomicUsize) {
        atomic.store(self.0, Ordering::Release);
    }
}

fn per_frame_slot_index(count: usize, frame: usize, index: usize) -> Option<usize> {
    let base = frame.checked_mul(count)?;
    base.checked_add(index)
}

pub(crate) fn buffer_id<T: BufferMarker>(index: usize) -> Option<BufferId> {
    T::ids_slice().get(index).map(BufferId::load)
}

pub(crate) fn per_frame_buffer_id<T: BufferMarker<Lifetime = PerFrame>>(
    frame: usize,
    index: usize,
) -> Option<BufferId> {
    T::ids_slice()
        .get(per_frame_slot_index(T::COUNT, frame, index)?)
        .map(BufferId::load)
}

pub(crate) fn image_id<T: ImageMarker>(index: usize) -> Option<ImageId> {
    T::ids_slice().get(index).map(ImageId::load)
}

pub(crate) fn per_frame_image_id<T: ImageMarker<Lifetime = PerFrame>>(
    frame: usize,
    index: usize,
) -> Option<ImageId> {
    T::ids_slice()
        .get(per_frame_slot_index(T::COUNT, frame, index)?)
        .map(ImageId::load)
}

pub(crate) fn image_view_id<T: ImageViewMarker>(index: usize) -> Option<ImageViewId> {
    T::ids_slice().get(index).map(ImageViewId::load)
}

pub(crate) fn per_frame_image_view_id<T: ImageViewMarker<Lifetime = PerFrame>>(
    frame: usize,
    index: usize,
) -> Option<ImageViewId> {
    T::ids_slice()
        .get(per_frame_slot_index(T::COUNT, frame, index)?)
        .map(ImageViewId::load)
}

pub(crate) fn descriptor_set_id<T: DescriptorSetMarker>() -> Option<DescriptorSetId> {
    T::ids_slice().first().map(DescriptorSetId::load)
}

pub(crate) fn per_frame_descriptor_set_id<T: DescriptorSetMarker>(
    frame: usize,
) -> Option<DescriptorSetId> {
    T::ids_slice().get(frame).map(DescriptorSetId::load)
}

pub(crate) fn sampler_id<T: SamplerMarker>() -> SamplerId {
    SamplerId::load(T::id())
}

pub(crate) fn descriptor_pool_id<T: DescriptorPoolMarker>() -> DescriptorPoolId {
    DescriptorPoolId::load(T::id())
}

pub(crate) fn graphics_pipeline_id<T: GraphicsPipelineMarker>() -> GraphicsPipelineId {
    GraphicsPipelineId::load(T::id())
}

pub(crate) fn compute_pipeline_id<T: ComputePipelineMarker>() -> ComputePipelineId {
    ComputePipelineId::load(T::id())
}

pub(crate) fn reusable_command_buffer_slot<M: ReusableCommandBufferMarker>(
    current_compute_frame: usize,
    index: usize,
) -> &'static AtomicUsize {
    assert!(
        index < M::COUNT,
        "Reusable command buffer index out of range"
    );

    let slot_index = per_frame_slot_index(M::COUNT, current_compute_frame, index)
        .expect("Reusable command buffer slot index overflowed");

    M::ids_slice()
        .get(slot_index)
        .expect("Reusable command buffer marker storage is too small")
}
