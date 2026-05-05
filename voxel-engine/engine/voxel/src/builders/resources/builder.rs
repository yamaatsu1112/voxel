use crate::builders::resources::buffers::*;
use crate::builders::resources::images::*;
use crate::rendering::renderer::Renderer;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct ResourceBuilder;

impl ResourceBuilder {
    pub fn build_renderer_resources(renderer: &mut Renderer<VulkanRenderer>) {
        create_fullscreen_vertex_buffer(renderer);
        create_index_buffer(renderer);
        create_leaf_edit_command_buffers(renderer);
        create_destroy_leaf_edit_command_buffers(renderer);
        create_command_count_buffers(renderer);
        create_temporary_compute_buffers(renderer);
        create_raycast_result_buffers(renderer);
        create_collision_buffers(renderer);
        create_dynamic_leaf_edit_command_buffers(renderer);
        create_dynamic_temporary_compute_buffers(renderer);
        create_gbuffer_per_frame_images(renderer);
    }
}
