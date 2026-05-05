mod command_resource_manager;
pub mod compute_executor;
mod context;
mod debug;
mod descriptors;
pub mod gpu_mutex;
#[cfg(feature = "gpu-profiling")]
pub(crate) mod gpu_timing;
pub mod pipeline_config;
mod pipeline_manager;
mod platforms;
pub mod record_context;
pub mod record_resource;
pub mod recordable;
pub mod resource_config;
pub mod resource_lifetime;
mod resource_manager;
mod resources;
pub(crate) mod staging_ring_buffer;
mod submit_group;
pub mod swapchain;
pub mod sync;
pub(crate) mod transfer_commands;
mod utils;
pub mod vulkan_renderer;
pub mod vulkan_renderer_core;
pub mod vk {
    pub use ash::vk::*;
}

pub use submit_group::ComputeSubmitGroup;
pub use submit_group::GraphicsSubmitGroup;
pub(crate) use submit_group::*;
