use ash::vk;

use crate::builders::configs::buffer_configs::compute::{MAX_LEAF_EDIT_COMMANDS, NUM_SVO_DATA};
use crate::vulkan::resource_config::{BufferConfig, Fixed};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NodeRequest {
    pub node_index: u32,
    pub child_index: u32,
}

engine_macro::define_buffer! {
    pub struct AllocationRequestBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<NodeRequest>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    pub struct UniqueRequestBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<NodeRequest>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    pub struct DynamicAllocationRequestBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<NodeRequest>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    pub struct DynamicUniqueRequestBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = NUM_SVO_DATA;
    config = BufferConfig {
        size: (std::mem::size_of::<NodeRequest>() * MAX_LEAF_EDIT_COMMANDS) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

pub(crate) use crate::builders::configs::buffer_configs::{
    DestroyIndirectDispatchBuffer, DynamicIdSVOCountersOriginalBuffer,
    DynamicIdSVOFreeLeafIndicesOriginalBuffer, DynamicIdSVOFreeNodeIndicesOriginalBuffer,
    DynamicIdSVOLeavesOriginalBuffer, DynamicIdSVOLeavesPerFrameBuffer,
    DynamicIdSVONodesOriginalBuffer, DynamicIdSVONodesPerFrameBuffer,
    DynamicIndirectDispatchBuffer, DynamicSVOCountersOriginalBuffer,
    DynamicSVOFreeLeafIndicesOriginalBuffer, DynamicSVOFreeNodeIndicesOriginalBuffer,
    DynamicSVOLeavesOriginalBuffer, DynamicSVOLeavesPerFrameBuffer, DynamicSVONodesOriginalBuffer,
    DynamicSVONodesPerFrameBuffer, ID_BIT_PLANES, IdSVOCountersOriginalBuffer,
    IdSVOFreeLeafIndicesOriginalBuffer, IdSVOFreeNodeIndicesOriginalBuffer,
    IdSVOLeavesOriginalBuffer, IdSVOLeavesPerFrameBuffer, IdSVONodesOriginalBuffer,
    IdSVONodesPerFrameBuffer, IndirectDispatchBuffer, SVOCountersOriginalBuffer,
    SVOFreeLeafIndicesOriginalBuffer, SVOFreeNodeIndicesOriginalBuffer, SVOLeavesOriginalBuffer,
    SVOLeavesPerFrameBuffer, SVONodesOriginalBuffer, SVONodesPerFrameBuffer,
};

#[cfg(test)]
mod tests {
    use super::NodeRequest;

    #[test]
    fn node_request_layout_matches_shader_struct() {
        assert_eq!(std::mem::size_of::<NodeRequest>(), 8);
    }
}
