pub mod submit_groups;

pub use engine_common::recordables::BeginSwapchainRendering;
pub use engine_common::recordables::{EndRendering, EndRenderingPhase, SetDynamicStates};
pub use engine_ui::recordables::UIPipelineRecordable;
pub use engine_voxel::recordables::BeginColorGBufferRendering;
pub use engine_voxel::recordables::BeginDynamicColorGBufferRendering;
pub use engine_voxel::recordables::BeginDynamicGBufferRendering;
pub use engine_voxel::recordables::BeginGBufferRendering;
pub use engine_voxel::recordables::CollisionComputePipelineRecordable;
pub use engine_voxel::recordables::CollisionComputeSubmitGroup;
pub use engine_voxel::recordables::CopyDynamicSVOSubmitGroup;
pub use engine_voxel::recordables::CopySVOSubmitGroup;
pub use engine_voxel::recordables::DynamicGBufferColorWritePipelineRecordable;
pub use engine_voxel::recordables::DynamicGBufferWritePipelineRecordable;
pub use engine_voxel::recordables::DynamicVoxelComputeSubmitGroup;
pub use engine_voxel::recordables::DynamicVoxelPipelineMutex;
pub use engine_voxel::recordables::DynamicVoxelPipelineRecordable;
pub use engine_voxel::recordables::GBufferColorWritePipelineRecordable;
pub use engine_voxel::recordables::GBufferWritePipelineRecordable;
pub use engine_voxel::recordables::GraphicsTransferMutex;
pub use engine_voxel::recordables::MainPipelineRecordable;
pub use engine_voxel::recordables::RaycastComputePipelineRecordable;
pub use engine_voxel::recordables::RaycastComputeSubmitGroup;
pub use engine_voxel::recordables::TransitDynamicGBufferColorImage;
pub use engine_voxel::recordables::TransitDynamicGBufferImage;
pub use engine_voxel::recordables::TransitGBufferColorImage;
pub use engine_voxel::recordables::TransitGBufferImage;
pub use engine_voxel::recordables::VoxelComputeSubmitGroup;
pub use engine_voxel::recordables::VoxelDestroyPipelineRecordable;
pub use engine_voxel::recordables::VoxelPipelineMutex;
pub use engine_voxel::recordables::VoxelPipelineRecordable;
pub use submit_groups::*;

pub fn construct_render_graph_system(world: &mut crate::ecs::world::World) {
    let mut render_graph = world
        .get_resource_mut::<crate::rendering::render_graph::RenderGraph>()
        .unwrap();
    render_graph.add_pass(CopySVOSubmitGroup);
    render_graph.add_pass(CopyDynamicSVOSubmitGroup);
    render_graph.add_pass(RenderSubmitGroup);
}
