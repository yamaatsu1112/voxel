use crate::builders::configs::buffer_configs::compute::ID_BIT_PLANES;
use crate::voxel::svo::{
    SVO, SVO_LEAF_GPU_BYTE_SIZE, SVO_NODE_GPU_BYTE_SIZE, SVOBackend, SVOCounters,
};
use crate::vulkan::resource_config::{BufferConfig, Fixed};
use ash::vk;
use std::mem::size_of;

const SVO_GPU_NODES_BYTE_SIZE: u64 = SVO_NODE_GPU_BYTE_SIZE * SVO::MAX_NODE_COUNT as u64;
const SVO_GPU_LEAVES_BYTE_SIZE: u64 = SVO_LEAF_GPU_BYTE_SIZE * SVO::MAX_LEAF_COUNT as u64;
const SVO_GPU_COUNTERS_BYTE_SIZE: u64 = size_of::<SVOCounters>() as u64;
const SVO_GPU_FREE_NODE_INDICES_BYTE_SIZE: u64 =
    size_of::<u32>() as u64 * SVO::MAX_NODE_COUNT as u64;
const SVO_GPU_FREE_LEAF_INDICES_BYTE_SIZE: u64 =
    size_of::<u32>() as u64 * SVO::MAX_LEAF_COUNT as u64;

engine_macro::define_buffer! {
    /// Original SVO node buffer updated by compute shader and copied to the fragment path
    pub struct SVONodesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_NODES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Per-frame SVO node buffer used by graphics pipeline
    pub struct SVONodesPerFrameBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_NODES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original SVO leaf buffer updated by compute shader and copied to the fragment path
    pub struct SVOLeavesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_LEAVES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Per-frame SVO leaf buffer used by graphics pipeline
    pub struct SVOLeavesPerFrameBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_LEAVES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original SVO counter buffer updated by compute shader
    pub struct SVOCountersOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_COUNTERS_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original SVO free-node index buffer updated by compute shader
    pub struct SVOFreeNodeIndicesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_FREE_NODE_INDICES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original SVO free-leaf index buffer updated by compute shader
    pub struct SVOFreeLeafIndicesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_FREE_LEAF_INDICES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original ID SVO node buffer for each bit plane
    pub struct IdSVONodesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_NODES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Per-frame ID SVO node buffer for each bit plane
    pub struct IdSVONodesPerFrameBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_NODES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original ID SVO leaf buffer for each bit plane
    pub struct IdSVOLeavesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_LEAVES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Per-frame ID SVO leaf buffer for each bit plane
    pub struct IdSVOLeavesPerFrameBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_LEAVES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original ID SVO counter buffer for each bit plane
    pub struct IdSVOCountersOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_COUNTERS_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original ID SVO free-node index buffer for each bit plane
    pub struct IdSVOFreeNodeIndicesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_FREE_NODE_INDICES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Original ID SVO free-leaf index buffer for each bit plane
    pub struct IdSVOFreeLeafIndicesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_FREE_LEAF_INDICES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Raycast result buffer (host-visible for CPU readback)
    pub struct RaycastResultBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: std::mem::size_of::<[u32; 5]>() as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        properties: vk::MemoryPropertyFlags::from_raw(
            vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        ),
    };
}

engine_macro::define_buffer! {
    /// Dynamic SVO node buffer updated by compute shader and copied to the fragment path
    pub struct DynamicSVONodesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_NODES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic SVO per-frame node buffer
    pub struct DynamicSVONodesPerFrameBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_NODES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic SVO leaf buffer updated by compute shader and copied to the fragment path
    pub struct DynamicSVOLeavesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_LEAVES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic SVO per-frame leaf buffer
    pub struct DynamicSVOLeavesPerFrameBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_LEAVES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic SVO counter buffer updated by compute shader
    pub struct DynamicSVOCountersOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_COUNTERS_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic SVO free-node index buffer updated by compute shader
    pub struct DynamicSVOFreeNodeIndicesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_FREE_NODE_INDICES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic SVO free-leaf index buffer updated by compute shader
    pub struct DynamicSVOFreeLeafIndicesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = 1;
    config = BufferConfig {
        size: SVO_GPU_FREE_LEAF_INDICES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic ID SVO original node buffer
    pub struct DynamicIdSVONodesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_NODES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic ID SVO per-frame node buffer
    pub struct DynamicIdSVONodesPerFrameBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_NODES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic ID SVO original leaf buffer
    pub struct DynamicIdSVOLeavesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_LEAVES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic ID SVO per-frame leaf buffer
    pub struct DynamicIdSVOLeavesPerFrameBuffer;
    size = Fixed;
    lifetime = PerFrame;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_LEAVES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic ID SVO counter buffer for each bit plane
    pub struct DynamicIdSVOCountersOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_COUNTERS_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic ID SVO free-node index buffer for each bit plane
    pub struct DynamicIdSVOFreeNodeIndicesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_FREE_NODE_INDICES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

engine_macro::define_buffer! {
    /// Dynamic ID SVO free-leaf index buffer for each bit plane
    pub struct DynamicIdSVOFreeLeafIndicesOriginalBuffer;
    size = Fixed;
    lifetime = Persistent;
    count = ID_BIT_PLANES;
    config = BufferConfig {
        size: SVO_GPU_FREE_LEAF_INDICES_BYTE_SIZE,
        usage: vk::BufferUsageFlags::from_raw(
            vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
        ),
        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::resource_config::BufferMarker;

    #[test]
    fn split_svo_buffer_sizes_match_packed_offsets() {
        assert_eq!(SVONodesOriginalBuffer::CONFIG.size, SVO_GPU_NODES_BYTE_SIZE);
        assert_eq!(SVONodesPerFrameBuffer::CONFIG.size, SVO_GPU_NODES_BYTE_SIZE);
        assert_eq!(
            SVOLeavesOriginalBuffer::CONFIG.size,
            SVO_GPU_LEAVES_BYTE_SIZE
        );
        assert_eq!(
            SVOLeavesPerFrameBuffer::CONFIG.size,
            SVO_GPU_LEAVES_BYTE_SIZE
        );
        assert_eq!(
            SVOCountersOriginalBuffer::CONFIG.size,
            SVO_GPU_COUNTERS_BYTE_SIZE
        );
        assert_eq!(
            SVOFreeNodeIndicesOriginalBuffer::CONFIG.size,
            SVO_GPU_FREE_NODE_INDICES_BYTE_SIZE
        );
        assert_eq!(
            SVOFreeLeafIndicesOriginalBuffer::CONFIG.size,
            SVO_GPU_FREE_LEAF_INDICES_BYTE_SIZE
        );
    }
}
