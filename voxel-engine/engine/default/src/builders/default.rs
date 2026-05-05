use crate::rendering::commands::CreateDescriptorPoolCommand;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub fn create_descriptor_pool(renderer: &mut Renderer<VulkanRenderer>) {
    use crate::builders::configs::descriptor_set_configs::EngineDescriptorPool;
    use ash::vk;
    use engine_ui::builders::configs::pipeline_configs::{UI_POOL_SIZES, UI_SET_COUNT};
    use engine_voxel::builders::configs::pipeline_configs::{
        COLLISION_COMPUTE_POOL_SIZES, COLLISION_COMPUTE_SET_COUNT, COMPUTE_POOL_SIZES,
        COMPUTE_SET_COUNT, DESTROY_COMPUTE_POOL_SIZES, DESTROY_COMPUTE_SET_COUNT,
        DYNAMIC_GBUFFER_COLOR_WRITE_POOL_SIZES, DYNAMIC_GBUFFER_COLOR_WRITE_SET_COUNT,
        DYNAMIC_GBUFFER_WRITE_POOL_SIZES, DYNAMIC_GBUFFER_WRITE_SET_COUNT,
        DYNAMIC_VOXEL_POOL_SIZES, DYNAMIC_VOXEL_SET_COUNT, GBUFFER_COLOR_WRITE_POOL_SIZES,
        GBUFFER_COLOR_WRITE_SET_COUNT, MAIN_POOL_SIZES, MAIN_SET_COUNT, MAIN_STATIC_POOL_SIZES,
        MAIN_STATIC_SET_COUNT, PREPARE_DISPATCH_POOL_SIZES, PREPARE_DISPATCH_SET_COUNT,
        RAYCAST_COMPUTE_POOL_SIZES, RAYCAST_COMPUTE_SET_COUNT,
    };

    let pool_sizes = [
        MAIN_POOL_SIZES,
        MAIN_STATIC_POOL_SIZES,
        UI_POOL_SIZES,
        COMPUTE_POOL_SIZES,
        DESTROY_COMPUTE_POOL_SIZES,
        RAYCAST_COMPUTE_POOL_SIZES,
        COLLISION_COMPUTE_POOL_SIZES,
        GBUFFER_COLOR_WRITE_POOL_SIZES,
        DYNAMIC_VOXEL_POOL_SIZES,
        DYNAMIC_GBUFFER_WRITE_POOL_SIZES,
        DYNAMIC_GBUFFER_COLOR_WRITE_POOL_SIZES,
        PREPARE_DISPATCH_POOL_SIZES,
        PREPARE_DISPATCH_POOL_SIZES,
        PREPARE_DISPATCH_POOL_SIZES,
    ]
    .concat();

    renderer.add_command(CreateDescriptorPoolCommand::<EngineDescriptorPool>::new(
        pool_sizes,
        (MAIN_SET_COUNT
            + MAIN_STATIC_SET_COUNT
            + UI_SET_COUNT
            + COMPUTE_SET_COUNT
            + DESTROY_COMPUTE_SET_COUNT
            + RAYCAST_COMPUTE_SET_COUNT
            + COLLISION_COMPUTE_SET_COUNT
            + GBUFFER_COLOR_WRITE_SET_COUNT
            + DYNAMIC_VOXEL_SET_COUNT
            + DYNAMIC_GBUFFER_WRITE_SET_COUNT
            + DYNAMIC_GBUFFER_COLOR_WRITE_SET_COUNT
            + PREPARE_DISPATCH_SET_COUNT * 3) as u32,
        vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
    ));
}
