use std::marker::PhantomData;

use crate::rendering::renderer_command::{
    OneShot, RendererBackend, RendererCommand, TypedRendererCommand,
};
use crate::vulkan::resource_config::{ComputePipelineMarker, GraphicsPipelineMarker};

pub struct CreateGraphicsPipelineCommand<T>
where
    T: GraphicsPipelineMarker,
{
    _marker: PhantomData<T>,
}

impl<T> Default for CreateGraphicsPipelineCommand<T>
where
    T: GraphicsPipelineMarker,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CreateGraphicsPipelineCommand<T>
where
    T: GraphicsPipelineMarker,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for CreateGraphicsPipelineCommand<T>
where
    T: GraphicsPipelineMarker,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreateGraphicsPipelineCommand<T>
where
    T: GraphicsPipelineMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<T, R> RendererCommand<R> for CreateGraphicsPipelineCommand<T>
where
    T: GraphicsPipelineMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.create_graphics_pipeline::<T>();
    }
}

pub struct CreateComputePipelineCommand<T>
where
    T: ComputePipelineMarker,
{
    _marker: PhantomData<T>,
}

impl<T> Default for CreateComputePipelineCommand<T>
where
    T: ComputePipelineMarker,
{
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> CreateComputePipelineCommand<T>
where
    T: ComputePipelineMarker,
{
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> Clone for CreateComputePipelineCommand<T>
where
    T: ComputePipelineMarker,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreateComputePipelineCommand<T>
where
    T: ComputePipelineMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<T, R> RendererCommand<R> for CreateComputePipelineCommand<T>
where
    T: ComputePipelineMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.create_compute_pipeline::<T>();
    }
}
