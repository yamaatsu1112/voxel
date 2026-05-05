use crate::rendering::{GpuMutex, NoMutex};
use crate::vulkan::{ComputeSubmitGroup, GraphicsSubmitGroup};

use super::{
    CollisionComputePipelineRecordable, CopyDynamicSVO, CopySVO, DynamicVoxelPipelineRecordable,
    RaycastComputePipelineRecordable, VoxelDestroyPipelineRecordable, VoxelPipelineRecordable,
};

pub struct VoxelPipelineMutex;

impl GpuMutex for VoxelPipelineMutex {}

pub struct DynamicVoxelPipelineMutex;

impl GpuMutex for DynamicVoxelPipelineMutex {}

pub struct GraphicsTransferMutex;

impl GpuMutex for GraphicsTransferMutex {}

#[derive(Clone)]
pub struct CopySVOSubmitGroup;

impl GraphicsSubmitGroup for CopySVOSubmitGroup {
    type Mutex = (VoxelPipelineMutex, GraphicsTransferMutex);
    type Recordables = CopySVO;
}

#[derive(Clone)]
pub struct CopyDynamicSVOSubmitGroup;

impl GraphicsSubmitGroup for CopyDynamicSVOSubmitGroup {
    type Mutex = DynamicVoxelPipelineMutex;
    type Recordables = CopyDynamicSVO;
}

pub struct VoxelComputeSubmitGroup;

impl ComputeSubmitGroup for VoxelComputeSubmitGroup {
    type Mutex = VoxelPipelineMutex;
    type Recordables = (VoxelDestroyPipelineRecordable, VoxelPipelineRecordable);
}

pub struct DynamicVoxelComputeSubmitGroup;

impl ComputeSubmitGroup for DynamicVoxelComputeSubmitGroup {
    type Mutex = DynamicVoxelPipelineMutex;
    type Recordables = DynamicVoxelPipelineRecordable;
}

pub struct RaycastComputeSubmitGroup;

impl ComputeSubmitGroup for RaycastComputeSubmitGroup {
    type Mutex = NoMutex;
    type Recordables = RaycastComputePipelineRecordable;
}

pub struct CollisionComputeSubmitGroup;

impl ComputeSubmitGroup for CollisionComputeSubmitGroup {
    type Mutex = NoMutex;
    type Recordables = CollisionComputePipelineRecordable;
}
