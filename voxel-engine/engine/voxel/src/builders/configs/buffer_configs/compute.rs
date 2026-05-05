use crate::voxel::command::LeafEditCommand;
use crate::vulkan::resource_config::{BufferConfig, Fixed};
use ash::vk;

/// Number of SVOData (1 for occupancy + 8 for ID bit planes)
pub const NUM_SVO_DATA: usize = 9;

/// Number of ID bit planes for voxel ID (8 bits)
pub const ID_BIT_PLANES: usize = 8;

/// Maximum number of leaf edit commands
pub const MAX_LEAF_EDIT_COMMANDS: usize = 65536;

pub const INDIRECT_DISPATCH_SLOTS_PER_SVO: usize = 3;
pub const INDIRECT_DISPATCH_COMMAND_COUNT: usize = NUM_SVO_DATA * INDIRECT_DISPATCH_SLOTS_PER_SVO;

engine_macro::define_buffer! {
    /// Device buffer for voxel command counts
    pub struct CommandCountBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * NUM_SVO_DATA) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Device buffer for destroy command counts
    pub struct DestroyCommandCountBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * NUM_SVO_DATA) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Device buffer for dynamic voxel command counts
    pub struct DynamicCommandCountBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * NUM_SVO_DATA) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Device buffer for voxel indirect dispatch commands
    pub struct IndirectDispatchBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: (std::mem::size_of::<vk::DispatchIndirectCommand>() * INDIRECT_DISPATCH_COMMAND_COUNT) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Device buffer for destroy indirect dispatch commands
    pub struct DestroyIndirectDispatchBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: (std::mem::size_of::<vk::DispatchIndirectCommand>() * INDIRECT_DISPATCH_COMMAND_COUNT) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Device buffer for dynamic voxel indirect dispatch commands
    pub struct DynamicIndirectDispatchBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: (std::mem::size_of::<vk::DispatchIndirectCommand>() * INDIRECT_DISPATCH_COMMAND_COUNT) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Device buffer for leaf edit commands (device-local)
    pub struct LeafEditCommandBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<LeafEditCommand>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Device buffer for destroy leaf edit commands (device-local)
    pub struct DestroyLeafEditCommandBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<LeafEditCommand>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Flag buffer for compute pipeline
    pub struct FlagBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Prefix sum buffer for compute pipeline
    pub struct PrefixSumBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Request count buffer for compute pipeline
    pub struct RequestCountBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic leaf edit command buffer
    pub struct DynamicLeafEditCommandBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<LeafEditCommand>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic flag buffer
    pub struct DynamicFlagBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic prefix sum buffer
    pub struct DynamicPrefixSumBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic request count buffer
    pub struct DynamicRequestCountBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<u32>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}
