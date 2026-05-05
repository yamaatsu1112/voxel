use super::configs::{
    BufferConfig, DescriptorSetConfig, ImageConfig, ImageViewConfig, SamplerConfig,
};
use crate::vulkan::pipeline_config::{ComputePipelineConfig, GraphicsPipelineConfig};
use crate::vulkan::resource_lifetime::ResourceLifetime;
use std::sync::atomic::AtomicUsize;

/// Marker trait for descriptor pool IDs.
pub trait DescriptorPoolMarker: 'static {
    fn id() -> &'static AtomicUsize;
}

/// Trait that associates an image marker type with its configuration.
pub trait ImageMarker: 'static {
    type Lifetime: ResourceLifetime;
    const COUNT: usize;
    const CONFIG: ImageConfig;

    /// Returns a reference to the flat AtomicUsize storage for ImageIds.
    ///
    /// For Persistent images: length is COUNT
    /// For PerFrame images: length is COUNT * MAX_FRAMES_IN_FLIGHT
    /// (PerFrame layout: `[frame0_img0, frame0_img1, ..., frame1_img0, ...]`)
    fn ids_slice() -> &'static [AtomicUsize];
}

/// Trait that associates an image view marker type with its source image marker type.
pub trait ImageViewMarker: 'static {
    /// The image marker type that this view is associated with.
    type Image: ImageMarker;
    type Lifetime: ResourceLifetime;
    const COUNT: usize;
    const CONFIG: ImageViewConfig;

    /// Returns a reference to the flat AtomicUsize storage for ImageViewIds.
    ///
    /// For Persistent image views: length is COUNT
    /// For PerFrame image views: length is COUNT * MAX_FRAMES_IN_FLIGHT
    fn ids_slice() -> &'static [AtomicUsize];
}

/// Trait that associates a descriptor set marker type with its configuration.
pub trait DescriptorSetMarker: 'static {
    type Lifetime: ResourceLifetime;
    type Pool: DescriptorPoolMarker;
    const CONFIG: DescriptorSetConfig;
    fn ids_slice() -> &'static [AtomicUsize];
}

/// Trait that associates a sampler marker type with its configuration.
pub trait SamplerMarker: 'static {
    const CONFIG: SamplerConfig;
    fn id() -> &'static AtomicUsize;
}

/// Marker trait for buffer size classification.
pub trait BufferSize: 'static {}

/// Fixed-size buffer.
pub struct Fixed;
impl BufferSize for Fixed {}

/// Trait that associates a buffer marker type with its configuration.
pub trait BufferMarker: 'static {
    type Size: BufferSize;
    type Lifetime: ResourceLifetime;
    const COUNT: usize;
    const CONFIG: BufferConfig;

    /// Returns a reference to the flat AtomicUsize storage for BufferIds.
    ///
    /// For Persistent buffers: length is COUNT
    /// For PerFrame buffers: length is COUNT * MAX_FRAMES_IN_FLIGHT
    /// (PerFrame layout: `[frame0_buf0, frame0_buf1, ..., frame1_buf0, ...]`)
    fn ids_slice() -> &'static [AtomicUsize];
}

/// Trait that associates a reusable command buffer marker type with its storage.
pub trait ReusableCommandBufferMarker: 'static {
    const COUNT: usize;

    /// Returns a reference to the flat AtomicUsize storage for reusable command buffer indices.
    ///
    /// Layout: `[frame0_cb0, frame0_cb1, ..., frame1_cb0, ...]`
    fn ids_slice() -> &'static [AtomicUsize];
}

/// Trait that associates a graphics pipeline marker type with its configuration.
pub trait GraphicsPipelineMarker: 'static {
    const CONFIG: GraphicsPipelineConfig;
    fn id() -> &'static AtomicUsize;
}

/// Trait that associates a compute pipeline marker type with its configuration.
pub trait ComputePipelineMarker: 'static {
    const CONFIG: ComputePipelineConfig;
    fn id() -> &'static AtomicUsize;
}
