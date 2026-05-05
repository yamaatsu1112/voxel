pub use engine_app::*;
pub use engine_macro::*;

pub use engine_default::voxel;
pub use engine_default::voxel::CommandOffsetResource;
pub use engine_default::voxel::DynamicVoxelObjectId;
pub use engine_default::voxel::GpuCommandOffsets;
pub use engine_default::voxel::NUM_SVO_DATA;
pub use engine_default::voxel::VOXEL_SIZE;
pub use engine_default::voxel::VoxelCommandBuffer;
pub use engine_default::voxel::VoxelDestroyCommandBuffer;
pub use engine_default::voxel::VoxelObjectData;
pub use engine_default::voxel::VoxelObjectId;
pub use engine_default::voxel::VoxelObjectManager;
pub use engine_default::voxel::VoxelRaycastData;

pub use engine_default::builders;
pub use engine_default::recordables;
pub use engine_default::setup;
pub use engine_default::ui_systems;

pub use engine_default::Button;
pub use engine_default::Camera;
pub use engine_default::CameraRenderData;
pub use engine_default::CameraRenderItem;
pub use engine_default::CameraTransformData;
pub use engine_default::Collider;
pub use engine_default::CollisionComputePipelineRecordable;
pub use engine_default::CollisionComputeSubmitGroup;
pub use engine_default::CollisionInputBuffer;
pub use engine_default::CollisionResultBuffer;
pub use engine_default::DynamicVoxelComputeSubmitGroup;
pub use engine_default::DynamicVoxelObject;
pub use engine_default::DynamicVoxelPipelineMutex;
pub use engine_default::DynamicVoxelPipelineRecordable;
pub use engine_default::GraphicsTransferMutex;
pub use engine_default::LocalTransform;
pub use engine_default::Parent;
pub use engine_default::RaycastComputePipelineRecordable;
pub use engine_default::RaycastComputeSubmitGroup;
pub use engine_default::Transform;
pub use engine_default::UI;
pub use engine_default::VoxelComputeSubmitGroup;
pub use engine_default::VoxelDestroyPipelineRecordable;
pub use engine_default::VoxelPipelineMutex;
pub use engine_default::VoxelPipelineRecordable;
pub use engine_default::WorldLinkAllocator;
pub use engine_default::WorldLinkId;
pub use engine_default::ensure_world_link_id;

pub fn init_logging() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();
}
