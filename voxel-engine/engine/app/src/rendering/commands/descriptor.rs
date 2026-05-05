use std::marker::PhantomData;

use ash::vk;

use crate::rendering::renderer_command::{
    OneShot, RendererBackend, RendererCommand, TypedRendererCommand,
};
use crate::vulkan::resource_config::{DescriptorPoolMarker, DescriptorSetMarker, SamplerMarker};

pub struct CreateDescriptorPoolCommand<P>
where
    P: DescriptorPoolMarker,
{
    pub pool_sizes: Vec<vk::DescriptorPoolSize>,
    pub max_sets: u32,
    pub flags: vk::DescriptorPoolCreateFlags,
    _marker: PhantomData<P>,
}

impl<P> CreateDescriptorPoolCommand<P>
where
    P: DescriptorPoolMarker,
{
    pub fn new(
        pool_sizes: Vec<vk::DescriptorPoolSize>,
        max_sets: u32,
        flags: vk::DescriptorPoolCreateFlags,
    ) -> Self {
        Self {
            pool_sizes,
            max_sets,
            flags,
            _marker: PhantomData,
        }
    }
}

impl<P> Clone for CreateDescriptorPoolCommand<P>
where
    P: DescriptorPoolMarker,
{
    fn clone(&self) -> Self {
        Self {
            pool_sizes: self.pool_sizes.clone(),
            max_sets: self.max_sets,
            flags: self.flags,
            _marker: PhantomData,
        }
    }
}

impl<P, R> TypedRendererCommand<R> for CreateDescriptorPoolCommand<P>
where
    P: DescriptorPoolMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<P, R> RendererCommand<R> for CreateDescriptorPoolCommand<P>
where
    P: DescriptorPoolMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.create_descriptor_pool::<P>(&self.pool_sizes, self.max_sets, self.flags);
    }
}

pub struct CreateDescriptorSetsCommand<T>
where
    T: DescriptorSetMarker,
{
    _marker: PhantomData<T>,
}

impl<T> CreateDescriptorSetsCommand<T>
where
    T: DescriptorSetMarker,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for CreateDescriptorSetsCommand<T>
where
    T: DescriptorSetMarker,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for CreateDescriptorSetsCommand<T>
where
    T: DescriptorSetMarker,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreateDescriptorSetsCommand<T>
where
    T: DescriptorSetMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<T, R> RendererCommand<R> for CreateDescriptorSetsCommand<T>
where
    T: DescriptorSetMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.create_descriptor_sets::<T>();
    }
}

pub struct CreateSamplerCommand<T>
where
    T: SamplerMarker,
{
    _marker: PhantomData<T>,
}

impl<T> Default for CreateSamplerCommand<T>
where
    T: SamplerMarker,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CreateSamplerCommand<T>
where
    T: SamplerMarker,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for CreateSamplerCommand<T>
where
    T: SamplerMarker,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreateSamplerCommand<T>
where
    T: SamplerMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<T, R> RendererCommand<R> for CreateSamplerCommand<T>
where
    T: SamplerMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.create_sampler::<T>();
    }
}
