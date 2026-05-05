pub mod ecs {
    pub use engine_ecs::*;
}

pub mod game_loop;
pub mod rendering;
pub mod vulkan;
pub mod window;

mod app;
pub mod input_event_queue;
mod voxel_engine;

pub use ecs::component::Component;
pub use ecs::entity::Entity;
pub use ecs::query::Query;
pub use ecs::query::QueryExt;
pub use ecs::resource::{Res, ResMut};
pub use ecs::scene::Scene;
pub use ecs::scene::SceneResource;
pub use ecs::scene::Scenes;
pub use ecs::world::World;
pub use voxel_engine::VoxelEngine;

pub use rendering::interpolation::{Interpolatable, Interpolated};
pub use rendering::render_world_renderer::RenderWorldRenderer;
pub use rendering::typed_channel::{GameChannel, Received, RenderChannel};
pub use rendering::{
    WorldLinkAllocator, WorldLinkEntry, WorldLinkId, WorldLinkRegistry, ensure_world_link_id,
};

pub use game_loop::DeltaTime;
pub use window::{Window, WindowSize};

pub use engine_input::InputState;
pub use engine_input::Key;
pub use engine_input::MouseButton;

pub use engine_types;

pub use rendering::commands::{
    BindBufferCommand, BindImageCommand, CreateBuffersCommand, CreateComputePipelineCommand,
    CreateDescriptorPoolCommand, CreateDescriptorSetsCommand, CreateImageViewsCommand,
    CreateImagesCommand, TransitionLayoutCommand, UploadToBufferCommand, UploadToImageCommand,
};
pub use rendering::executor::Executor;
pub use rendering::render_graph::RenderGraph;
pub use rendering::renderer::Renderer;

pub use rendering::NoMutex;
pub use vulkan::ComputeSubmitGroup;
pub use vulkan::pipeline_config::ComputePipelineConfig;
pub use vulkan::record_context::{
    IndexType, SecondaryCommandBufferHandle, ShaderStageFlags, VkComputeRecordContext,
    VkGraphicsRecordContext, VkSecondaryRecordContext,
};
pub use vulkan::recordable::VkComputeRecordable;
pub use vulkan::resource_config::{
    ComputePipelineMarker, DescriptorPoolMarker, DescriptorSetConfig, DescriptorSetMarker,
};
pub use vulkan::resource_lifetime::{PerFrame, Persistent};
pub use vulkan::vk;

pub use engine_input;
pub use engine_macro;
pub use engine_macro::Component;
pub use engine_macro::Scenes;
