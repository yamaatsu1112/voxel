use std::marker::PhantomData;

use ash::vk;

use crate::rendering::renderer_command::{
    OneShot, PerFrame, RendererBackend, RendererCommand, TypedRendererCommand,
};
use crate::vulkan::resource_config::{
    BufferMarker, DescriptorSetMarker, ImageViewMarker, SamplerMarker,
};
use crate::vulkan::resource_lifetime::{PerFrame as PerFrameLifetime, Persistent};

pub struct BindBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = Persistent>,
    B: BufferMarker<Lifetime = Persistent>,
{
    pub binding: u32,
    pub array_index: u32,
    pub buffer_index: usize,
    pub count: usize,
    pub descriptor_type: vk::DescriptorType,
    _marker: PhantomData<(D, B)>,
}

impl<D, B> BindBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = Persistent>,
    B: BufferMarker<Lifetime = Persistent>,
{
    pub fn new(
        binding: u32,
        array_index: u32,
        buffer_index: usize,
        count: usize,
        descriptor_type: vk::DescriptorType,
    ) -> Self {
        Self {
            binding,
            array_index,
            buffer_index,
            count,
            descriptor_type,
            _marker: PhantomData,
        }
    }
}

impl<D, B> Clone for BindBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = Persistent>,
    B: BufferMarker<Lifetime = Persistent>,
{
    fn clone(&self) -> Self {
        Self {
            binding: self.binding,
            array_index: self.array_index,
            buffer_index: self.buffer_index,
            count: self.count,
            descriptor_type: self.descriptor_type,
            _marker: PhantomData,
        }
    }
}

impl<D, B, R> TypedRendererCommand<R> for BindBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = Persistent> + Send + Sync + 'static,
    B: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<D, B, R> RendererCommand<R> for BindBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = Persistent> + Send + Sync + 'static,
    B: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.bind_buffer::<D, B>(
            self.binding,
            self.array_index,
            self.buffer_index,
            self.count,
            self.descriptor_type,
        );
    }
}

pub struct BindPerFrameBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime>,
    B: BufferMarker<Lifetime = PerFrameLifetime>,
{
    pub binding: u32,
    pub array_index: u32,
    pub buffer_index: usize,
    pub count: usize,
    pub descriptor_type: vk::DescriptorType,
    _marker: PhantomData<(D, B)>,
}

impl<D, B> BindPerFrameBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime>,
    B: BufferMarker<Lifetime = PerFrameLifetime>,
{
    pub fn new(
        binding: u32,
        array_index: u32,
        buffer_index: usize,
        count: usize,
        descriptor_type: vk::DescriptorType,
    ) -> Self {
        Self {
            binding,
            array_index,
            buffer_index,
            count,
            descriptor_type,
            _marker: PhantomData,
        }
    }
}

impl<D, B> Clone for BindPerFrameBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime>,
    B: BufferMarker<Lifetime = PerFrameLifetime>,
{
    fn clone(&self) -> Self {
        Self {
            binding: self.binding,
            array_index: self.array_index,
            buffer_index: self.buffer_index,
            count: self.count,
            descriptor_type: self.descriptor_type,
            _marker: PhantomData,
        }
    }
}

impl<D, B, R> TypedRendererCommand<R> for BindPerFrameBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    B: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = PerFrame;
}

impl<D, B, R> RendererCommand<R> for BindPerFrameBufferCommand<D, B>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    B: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.bind_per_frame_buffer::<D, B>(
            self.binding,
            self.array_index,
            self.buffer_index,
            self.count,
            self.descriptor_type,
        );
    }
}

pub struct BindImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = Persistent>,
    V: ImageViewMarker<Lifetime = Persistent>,
    S: SamplerMarker,
{
    pub binding: u32,
    pub array_index: u32,
    pub image_index: usize,
    pub count: usize,
    _marker: PhantomData<(D, V, S)>,
}

impl<D, V, S> BindImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = Persistent>,
    V: ImageViewMarker<Lifetime = Persistent>,
    S: SamplerMarker,
{
    pub fn new(binding: u32, array_index: u32, image_index: usize, count: usize) -> Self {
        Self {
            binding,
            array_index,
            image_index,
            count,
            _marker: PhantomData,
        }
    }
}

impl<D, V, S> Clone for BindImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = Persistent>,
    V: ImageViewMarker<Lifetime = Persistent>,
    S: SamplerMarker,
{
    fn clone(&self) -> Self {
        Self {
            binding: self.binding,
            array_index: self.array_index,
            image_index: self.image_index,
            count: self.count,
            _marker: PhantomData,
        }
    }
}

impl<D, V, S, R> TypedRendererCommand<R> for BindImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = Persistent> + Send + Sync + 'static,
    V: ImageViewMarker<Lifetime = Persistent> + Send + Sync + 'static,
    S: SamplerMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<D, V, S, R> RendererCommand<R> for BindImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = Persistent> + Send + Sync + 'static,
    V: ImageViewMarker<Lifetime = Persistent> + Send + Sync + 'static,
    S: SamplerMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.bind_image::<D, V, S>(
            self.binding,
            self.array_index,
            self.image_index,
            self.count,
        );
    }
}

pub struct BindPerFrameImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime>,
    V: ImageViewMarker<Lifetime = PerFrameLifetime>,
    S: SamplerMarker,
{
    pub binding: u32,
    pub array_index: u32,
    pub image_index: usize,
    pub count: usize,
    pub _marker: PhantomData<(D, V, S)>,
}

impl<D, V, S> BindPerFrameImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime>,
    V: ImageViewMarker<Lifetime = PerFrameLifetime>,
    S: SamplerMarker,
{
    pub fn new(binding: u32, array_index: u32, image_index: usize, count: usize) -> Self {
        Self {
            binding,
            array_index,
            image_index,
            count,
            _marker: PhantomData,
        }
    }
}

impl<D, V, S> Clone for BindPerFrameImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime>,
    V: ImageViewMarker<Lifetime = PerFrameLifetime>,
    S: SamplerMarker,
{
    fn clone(&self) -> Self {
        Self {
            binding: self.binding,
            array_index: self.array_index,
            image_index: self.image_index,
            count: self.count,
            _marker: PhantomData,
        }
    }
}

impl<D, V, S, R> TypedRendererCommand<R> for BindPerFrameImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    V: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    S: SamplerMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = PerFrame;
}

impl<D, V, S, R> RendererCommand<R> for BindPerFrameImageCommand<D, V, S>
where
    D: DescriptorSetMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    V: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    S: SamplerMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.bind_per_frame_image::<D, V, S>(
            self.binding,
            self.array_index,
            self.image_index,
            self.count,
        );
    }
}
