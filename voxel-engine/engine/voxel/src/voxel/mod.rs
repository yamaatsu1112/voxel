pub mod command;
pub mod command_buffer;
mod gpu_command_offsets;
pub mod manager;
pub mod object;
pub mod raycast_data;
pub mod svo;

// Re-export SVO constants for convenience
pub use svo::VOXEL_SIZE;
#[allow(unused_imports)]
pub use svo::WORLD_VOXEL_COUNT;
#[allow(unused_imports)]
pub use svo::{LEAF_VOXEL_COUNT, LEAF_VOXEL_COUNT_EXP};

// Re-export VoxelRaycastData for public use
pub use raycast_data::VoxelRaycastData;

// Re-export ID types for public use
pub use object::DynamicVoxelObjectId;
pub use object::VoxelObjectData;
pub use object::VoxelObjectId;

// Re-export VoxelObjectManager for public use
pub use command_buffer::VoxelCommandBuffer;
pub use command_buffer::VoxelDestroyCommandBuffer;
pub use gpu_command_offsets::{CommandOffsetResource, GpuCommandOffsets};
pub use manager::VoxelObjectManager;

// Re-export NUM_SVO_DATA constant
pub use manager::NUM_SVO_DATA;

pub use raycast_data::RawVoxelRaycastData;
