use crate::vulkan::command_resource_manager::{CommandPoolId, CommandResourceManager};
use crate::vulkan::context::VulkanContext;
use crate::vulkan::vulkan_renderer_core::{MAX_COMPUTE_FRAMES_IN_FLIGHT, MAX_FRAMES_IN_FLIGHT};
use ash::vk;

/// Aggregated command resources required by the renderer.
/// Graphics resources are per-frame (indexed by current_frame).
/// Compute resources are per-frame (indexed by current_compute_frame).
pub struct RendererCommandResources {
    pub graphics_pool_ids: Vec<CommandPoolId>,
    pub graphics_transfer_pool_ids: Vec<CommandPoolId>,
}

pub struct ComputeCommandResources {
    pub compute_secondary_command_pools: [CommandPoolId; MAX_COMPUTE_FRAMES_IN_FLIGHT],
    pub reusable_compute_pools: [CommandPoolId; MAX_COMPUTE_FRAMES_IN_FLIGHT],
    pub compute_pool_ids: [CommandPoolId; MAX_COMPUTE_FRAMES_IN_FLIGHT],
    pub compute_transfer_pool_ids: [CommandPoolId; MAX_COMPUTE_FRAMES_IN_FLIGHT],
}

pub struct CommandResourceBuilder;

impl CommandResourceBuilder {
    /// Build all renderer command resources at initialization time.
    /// This function centralizes command pool and buffer creation.
    pub fn build_renderer_command_resources(
        context: &VulkanContext,
        command_resource_manager: &mut CommandResourceManager,
    ) -> RendererCommandResources {
        // Create per-frame command pools for graphics
        let mut graphics_pool_ids = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let graphics_pool_id = command_resource_manager.create_command_pool(
                context.family_indices.graphics_family.unwrap(),
                vk::CommandPoolCreateFlags::empty(),
            );
            graphics_pool_ids.push(graphics_pool_id);
        }

        let mut graphics_transfer_pool_ids = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let transfer_pool_id = command_resource_manager.create_command_pool(
                context.family_indices.transfer_family.unwrap(),
                vk::CommandPoolCreateFlags::empty(),
            );
            graphics_transfer_pool_ids.push(transfer_pool_id);
        }

        RendererCommandResources {
            graphics_pool_ids,
            graphics_transfer_pool_ids,
        }
    }

    pub fn build_compute_command_resources(
        context: &VulkanContext,
        command_resource_manager: &mut CommandResourceManager,
    ) -> ComputeCommandResources {
        let compute_secondary_command_pools = std::array::from_fn(|_| {
            command_resource_manager.create_command_pool(
                context.family_indices.compute_family.unwrap(),
                vk::CommandPoolCreateFlags::empty(),
            )
        });

        let reusable_compute_pools = std::array::from_fn(|_| {
            command_resource_manager.create_command_pool(
                context.family_indices.compute_family.unwrap(),
                vk::CommandPoolCreateFlags::empty(),
            )
        });

        let compute_pool_ids = std::array::from_fn(|_| {
            command_resource_manager.create_command_pool(
                context.family_indices.compute_family.unwrap(),
                vk::CommandPoolCreateFlags::empty(),
            )
        });

        let compute_transfer_pool_ids = std::array::from_fn(|_| {
            command_resource_manager.create_command_pool(
                context.family_indices.transfer_family.unwrap(),
                vk::CommandPoolCreateFlags::empty(),
            )
        });

        ComputeCommandResources {
            compute_secondary_command_pools,
            reusable_compute_pools,
            compute_pool_ids,
            compute_transfer_pool_ids,
        }
    }
}
