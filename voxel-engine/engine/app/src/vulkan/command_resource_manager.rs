use ash::vk;
use std::marker::PhantomData;
use std::ptr;

/// Type-safe identifier for a command pool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommandPoolId(usize);

/// Information about a command pool resource
struct CommandPoolInfo {
    command_pool: vk::CommandPool,
    // Store primary command buffers allocated from this pool
    primary_command_buffers: Vec<vk::CommandBuffer>,
    // Store secondary command buffers allocated from this pool
    secondary_command_buffers: Vec<vk::CommandBuffer>,
    // Index of next command buffer to return for primary buffers
    next_primary_index: usize,
    // Index of next command buffer to return for secondary buffers
    next_secondary_index: usize,
}

/// Manages Vulkan command pools and command buffers
pub struct CommandResourceManager {
    device: ash::Device,
    command_pools: Vec<Option<CommandPoolInfo>>,
    free_pool_indices: Vec<usize>,
}

impl CommandResourceManager {
    /// Create a new CommandResourceManager
    pub fn new(device: ash::Device) -> Self {
        Self {
            device,
            command_pools: Vec::new(),
            free_pool_indices: Vec::new(),
        }
    }

    /// Create a command pool and return its ID
    pub fn create_command_pool(
        &mut self,
        queue_family_index: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> CommandPoolId {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags,
            queue_family_index,
            _marker: PhantomData,
        };

        let command_pool = unsafe {
            self.device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create command pool")
        };

        let pool_info = CommandPoolInfo {
            command_pool,
            primary_command_buffers: Vec::new(),
            secondary_command_buffers: Vec::new(),
            next_primary_index: 0,
            next_secondary_index: 0,
        };

        // Reuse free index if available, otherwise append
        let pool_index = if let Some(free_index) = self.free_pool_indices.pop() {
            self.command_pools[free_index] = Some(pool_info);
            free_index
        } else {
            let index = self.command_pools.len();
            self.command_pools.push(Some(pool_info));
            index
        };

        CommandPoolId(pool_index)
    }

    /// Get command pool by ID
    pub fn get_command_pool(&self, pool_id: CommandPoolId) -> Option<vk::CommandPool> {
        self.command_pools
            .get(pool_id.0)?
            .as_ref()
            .map(|info| info.command_pool)
    }

    pub fn get_command_buffer_in_pool(
        &self,
        pool_id: CommandPoolId,
        level: vk::CommandBufferLevel,
        buffer_index: usize,
    ) -> Option<vk::CommandBuffer> {
        let pool_info = self.command_pools.get(pool_id.0)?.as_ref()?;

        let buffer_vec = match level {
            vk::CommandBufferLevel::PRIMARY => &pool_info.primary_command_buffers,
            vk::CommandBufferLevel::SECONDARY => &pool_info.secondary_command_buffers,
            _ => return None,
        };

        buffer_vec.get(buffer_index).copied()
    }

    /// Reset a command pool, which implicitly resets all command buffers allocated from it
    pub fn reset_command_pool(&mut self, pool_id: CommandPoolId) {
        if let Some(command_pool) = self.get_command_pool(pool_id) {
            unsafe {
                self.device
                    .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                    .expect("Failed to reset command pool");
            }
            // Reset indices when pool is reset
            if let Some(pool_info) = self
                .command_pools
                .get_mut(pool_id.0)
                .and_then(|opt| opt.as_mut())
            {
                pool_info.next_primary_index = 0;
                pool_info.next_secondary_index = 0;
            }
        }
    }

    /// Get a free command buffer from the specified pool.
    /// Returns a recordable command buffer from the pool.
    /// If no free buffers exist, a new one is allocated.
    pub fn get_free_command_buffer(
        &mut self,
        pool_id: CommandPoolId,
        level: vk::CommandBufferLevel,
    ) -> vk::CommandBuffer {
        self.get_free_command_buffer_with_index(pool_id, level).0
    }

    pub fn get_free_command_buffer_with_index(
        &mut self,
        pool_id: CommandPoolId,
        level: vk::CommandBufferLevel,
    ) -> (vk::CommandBuffer, usize) {
        let pool_index = pool_id.0;
        let pool_info = self
            .command_pools
            .get_mut(pool_index)
            .and_then(|opt| opt.as_mut())
            .expect("Command pool not found");

        let (next_index, buffer_vec) = match level {
            vk::CommandBufferLevel::PRIMARY => (
                &mut pool_info.next_primary_index,
                &mut pool_info.primary_command_buffers,
            ),
            vk::CommandBufferLevel::SECONDARY => (
                &mut pool_info.next_secondary_index,
                &mut pool_info.secondary_command_buffers,
            ),
            _ => panic!("Invalid command buffer level"),
        };

        // If we have a buffer available, return it
        if *next_index < buffer_vec.len() {
            let buffer_index = *next_index;
            let buffer = buffer_vec[buffer_index];
            *next_index += 1;
            (buffer, buffer_index)
        } else {
            // Otherwise allocate a new one
            let command_pool = pool_info.command_pool;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: ptr::null(),
                command_pool,
                level,
                command_buffer_count: 1,
                _marker: PhantomData,
            };

            let command_buffer = unsafe {
                self.device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .expect("Failed to allocate command buffer")[0]
            };

            let buffer_index = buffer_vec.len();
            buffer_vec.push(command_buffer);
            *next_index += 1;
            (command_buffer, buffer_index)
        }
    }

    /// Destroy a command pool and return its index to the free list
    #[allow(dead_code)]
    pub fn destroy_command_pool(&mut self, pool_id: CommandPoolId) {
        if let Some(pool_info_opt) = self.command_pools.get_mut(pool_id.0)
            && let Some(pool_info) = pool_info_opt.take()
        {
            unsafe {
                // Destroy all command buffers are implicitly freed when the pool is destroyed
                self.device
                    .destroy_command_pool(pool_info.command_pool, None);
            }
            // Add to free_indices for reuse
            self.free_pool_indices.push(pool_id.0);
        }
    }

    /// Explicitly clean up all command resources
    /// This should be called before destroying the device
    pub fn cleanup(&mut self) {
        unsafe {
            // Destroy all command pools (this also implicitly frees all command buffers)
            for pool_info in self.command_pools.iter().flatten() {
                self.device
                    .destroy_command_pool(pool_info.command_pool, None);
            }
            self.command_pools.clear();
            self.free_pool_indices.clear();
        }
    }
}
