use ash::vk;
use std::marker::PhantomData;
use std::ptr;

pub(crate) const STAGING_BUFFER_CAPACITY: u64 = 8 * 1024 * 1024;
const STAGING_ALIGNMENT: u64 = 4;

pub(crate) struct StagingAllocation {
    pub buffer: vk::Buffer,
    pub offset: u64,
}

pub(crate) struct StagingRingBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: *mut u8,
    capacity: u64,
    write_offset: u64,
    overflow_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
}

impl StagingRingBuffer {
    pub(crate) fn new(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        queue_family_indices: &[u32],
    ) -> Self {
        let (buffer, memory) = create_host_visible_buffer(
            device,
            physical_device,
            instance,
            queue_family_indices,
            STAGING_BUFFER_CAPACITY,
        );
        let mapped_ptr = unsafe {
            device
                .map_memory(
                    memory,
                    0,
                    STAGING_BUFFER_CAPACITY,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map staging ring buffer memory") as *mut u8
        };

        Self {
            buffer,
            memory,
            mapped_ptr,
            capacity: STAGING_BUFFER_CAPACITY,
            write_offset: 0,
            overflow_buffers: Vec::new(),
        }
    }

    pub(crate) fn allocate_and_write<D>(
        &mut self,
        data: &[D],
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        queue_family_indices: &[u32],
    ) -> StagingAllocation {
        let data_size = std::mem::size_of_val(data) as u64;
        let aligned_size = align_up(data_size, STAGING_ALIGNMENT);

        if let Some(offset) = self.try_allocate_offset(aligned_size) {
            if data_size > 0 {
                unsafe {
                    ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        self.mapped_ptr.add(offset as usize),
                        data_size as usize,
                    );
                }
            }
            return StagingAllocation {
                buffer: self.buffer,
                offset,
            };
        }

        let (buffer, memory) = create_host_visible_buffer(
            device,
            physical_device,
            instance,
            queue_family_indices,
            data_size,
        );
        if data_size > 0 {
            unsafe {
                let mapped_ptr = device
                    .map_memory(memory, 0, data_size, vk::MemoryMapFlags::empty())
                    .expect("Failed to map overflow staging buffer memory")
                    as *mut u8;
                ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    mapped_ptr,
                    data_size as usize,
                );
                device.unmap_memory(memory);
            }
        }
        self.overflow_buffers.push((buffer, memory));

        StagingAllocation { buffer, offset: 0 }
    }

    fn try_allocate_offset(&mut self, aligned_size: u64) -> Option<u64> {
        if self.write_offset + aligned_size > self.capacity {
            return None;
        }
        let offset = self.write_offset;
        self.write_offset += aligned_size;
        Some(offset)
    }

    pub(crate) fn reset(&mut self, device: &ash::Device) {
        self.write_offset = 0;
        for (buffer, memory) in self.overflow_buffers.drain(..) {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
        }
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device) {
        self.reset(device);
        unsafe {
            if !self.mapped_ptr.is_null() {
                device.unmap_memory(self.memory);
            }
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
        self.mapped_ptr = ptr::null_mut();
        self.buffer = vk::Buffer::null();
        self.memory = vk::DeviceMemory::null();
        self.write_offset = 0;
    }
}

fn create_host_visible_buffer(
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    instance: &ash::Instance,
    queue_family_indices: &[u32],
    size: u64,
) -> (vk::Buffer, vk::DeviceMemory) {
    let (sharing_mode, queue_family_index_count, queue_family_indices_ptr) =
        if queue_family_indices.len() > 1 {
            (
                vk::SharingMode::CONCURRENT,
                queue_family_indices.len() as u32,
                queue_family_indices.as_ptr(),
            )
        } else {
            (vk::SharingMode::EXCLUSIVE, 0, ptr::null())
        };

    let create_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::BufferCreateFlags::empty(),
        size,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        sharing_mode,
        queue_family_index_count,
        p_queue_family_indices: queue_family_indices_ptr,
        _marker: PhantomData,
    };

    let buffer = unsafe {
        device
            .create_buffer(&create_info, None)
            .expect("Failed to create staging buffer")
    };

    let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = find_memory_type(
        instance,
        physical_device,
        memory_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),
        allocation_size: memory_requirements.size,
        memory_type_index,
        _marker: PhantomData,
    };

    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocate staging buffer memory")
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("Failed to bind staging buffer memory")
    }

    (buffer, memory)
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for i in 0..memory_properties.memory_type_count {
        let memory_type = memory_properties.memory_types[i as usize];
        if (type_filter & (1 << i) != 0) && (memory_type.property_flags & properties == properties)
        {
            return i;
        }
    }
    panic!("Failed to find suitable staging buffer memory type");
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if value == 0 {
        return 0;
    }
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::{StagingRingBuffer, align_up};
    use ash::vk;
    use std::ptr;

    #[test]
    fn align_up_keeps_aligned_values() {
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(512, 256), 512);
    }

    #[test]
    fn align_up_rounds_up_unaligned_values() {
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(300, 256), 512);
    }

    #[test]
    fn align_up_handles_zero_size() {
        assert_eq!(align_up(0, 256), 0);
    }

    #[test]
    fn try_allocate_offset_advances_write_offset_within_capacity() {
        let mut ring = StagingRingBuffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped_ptr: ptr::null_mut(),
            capacity: 1024,
            write_offset: 0,
            overflow_buffers: Vec::new(),
        };

        let first = ring.try_allocate_offset(256);
        let second = ring.try_allocate_offset(512);

        assert_eq!(first, Some(0));
        assert_eq!(second, Some(256));
        assert_eq!(ring.write_offset, 768);
    }

    #[test]
    fn try_allocate_offset_returns_none_when_out_of_capacity() {
        let mut ring = StagingRingBuffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped_ptr: ptr::null_mut(),
            capacity: 1024,
            write_offset: 900,
            overflow_buffers: Vec::new(),
        };

        let allocation = ring.try_allocate_offset(256);

        assert_eq!(allocation, None);
        assert_eq!(ring.write_offset, 900);
    }
}
