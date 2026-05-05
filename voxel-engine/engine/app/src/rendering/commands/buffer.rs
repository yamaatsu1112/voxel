use std::marker::PhantomData;
use std::sync::Arc;

use crate::rendering::GpuMutex;
use crate::rendering::renderer_command::{
    OneShot, PerFrame, RendererBackend, RendererCommand, TypedRendererCommand,
};
use crate::vulkan::resource_config::BufferMarker;
use crate::vulkan::resource_lifetime::{PerFrame as PerFrameLifetime, Persistent};

pub struct CreateBuffersCommand<T>
where
    T: BufferMarker,
{
    _marker: PhantomData<T>,
}

impl<T> Default for CreateBuffersCommand<T>
where
    T: BufferMarker,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CreateBuffersCommand<T>
where
    T: BufferMarker,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for CreateBuffersCommand<T>
where
    T: BufferMarker,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreateBuffersCommand<T>
where
    T: BufferMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<T, R> RendererCommand<R> for CreateBuffersCommand<T>
where
    T: BufferMarker + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.create_buffers::<T>();
    }
}

pub struct UploadToBufferCommand<M, T, D>
where
    M: GpuMutex,
    T: BufferMarker<Lifetime = Persistent>,
{
    pub index: usize,
    pub data: UploadBufferData<D>,
    pub dst_byte_offset: u64,
    _marker: PhantomData<(M, T)>,
}

pub enum UploadBufferData<D> {
    Owned(Box<[D]>),
    SharedValue(Arc<D>),
}

impl<M, T, D> UploadToBufferCommand<M, T, D>
where
    M: GpuMutex,
    T: BufferMarker<Lifetime = Persistent>,
    D: Clone,
{
    pub fn new(index: usize, data: &[D], dst_byte_offset: u64) -> Self {
        Self {
            index,
            data: UploadBufferData::Owned(data.to_vec().into_boxed_slice()),
            dst_byte_offset,
            _marker: PhantomData,
        }
    }

    pub fn from_shared_value(index: usize, data: Arc<D>, dst_byte_offset: u64) -> Self {
        Self {
            index,
            data: UploadBufferData::SharedValue(data),
            dst_byte_offset,
            _marker: PhantomData,
        }
    }
}

impl<D: Clone> Clone for UploadBufferData<D> {
    fn clone(&self) -> Self {
        match self {
            Self::Owned(data) => Self::Owned(data.clone()),
            Self::SharedValue(data) => Self::SharedValue(Arc::clone(data)),
        }
    }
}

impl<M, T, D> Clone for UploadToBufferCommand<M, T, D>
where
    M: GpuMutex,
    T: BufferMarker<Lifetime = Persistent>,
    D: Clone,
{
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            data: self.data.clone(),
            dst_byte_offset: self.dst_byte_offset,
            _marker: PhantomData,
        }
    }
}

impl<M, T, D, R> TypedRendererCommand<R> for UploadToBufferCommand<M, T, D>
where
    M: GpuMutex + Send + Sync + 'static,
    T: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
    D: Clone + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<M, T, D, R> RendererCommand<R> for UploadToBufferCommand<M, T, D>
where
    M: GpuMutex + Send + Sync + 'static,
    T: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
    D: Clone + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        match &self.data {
            UploadBufferData::Owned(data) => {
                renderer.upload_to_buffer::<M, T, D>(self.index, data, self.dst_byte_offset);
            }
            UploadBufferData::SharedValue(data) => {
                renderer.upload_to_buffer::<M, T, D>(
                    self.index,
                    std::slice::from_ref(data.as_ref()),
                    self.dst_byte_offset,
                );
            }
        }
    }
}

pub struct UploadToPerFrameBufferCommand<M, T, D>
where
    M: GpuMutex,
    T: BufferMarker<Lifetime = PerFrameLifetime>,
{
    pub index: usize,
    pub data: Vec<D>,
    _marker: PhantomData<(M, T)>,
}

impl<M, T, D> UploadToPerFrameBufferCommand<M, T, D>
where
    M: GpuMutex,
    T: BufferMarker<Lifetime = PerFrameLifetime>,
    D: Clone,
{
    pub fn new(index: usize, data: &[D]) -> Self {
        Self {
            index,
            data: data.to_vec(),
            _marker: PhantomData,
        }
    }
}

impl<M, T, D> Clone for UploadToPerFrameBufferCommand<M, T, D>
where
    M: GpuMutex,
    T: BufferMarker<Lifetime = PerFrameLifetime>,
    D: Clone,
{
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            data: self.data.clone(),
            _marker: PhantomData,
        }
    }
}

impl<M, T, D, R> TypedRendererCommand<R> for UploadToPerFrameBufferCommand<M, T, D>
where
    M: GpuMutex + Send + Sync + 'static,
    T: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    D: Clone + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = PerFrame;
}

impl<M, T, D, R> RendererCommand<R> for UploadToPerFrameBufferCommand<M, T, D>
where
    M: GpuMutex + Send + Sync + 'static,
    T: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    D: Clone + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.upload_to_per_frame_buffer::<M, T, D>(self.index, &self.data);
    }
}

#[cfg(test)]
mod tests {
    use super::{UploadToBufferCommand, UploadToPerFrameBufferCommand};
    use crate::rendering::renderer_command::{RendererBackend, RendererCommand};
    use crate::rendering::{GpuMutex, NoMutex};
    use crate::vulkan::resource_config::{BufferConfig, BufferMarker, Fixed};
    use crate::vulkan::resource_lifetime::{PerFrame as PerFrameLifetime, Persistent};
    use crate::vulkan::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT;
    use ash::vk;
    use std::any::TypeId;

    use std::sync::atomic::AtomicUsize;

    struct TestBuffer;

    impl BufferMarker for TestBuffer {
        type Size = Fixed;
        type Lifetime = Persistent;
        const COUNT: usize = 1;
        const CONFIG: BufferConfig = BufferConfig {
            size: 16,
            usage: vk::BufferUsageFlags::empty(),
            properties: vk::MemoryPropertyFlags::empty(),
        };

        fn ids_slice() -> &'static [AtomicUsize] {
            // SAFETY: AtomicUsize and usize have identical memory layout.
            static IDS: [AtomicUsize; 1] = unsafe { std::mem::transmute([usize::MAX; 1]) };
            &IDS
        }
    }

    struct TestPerFrameBuffer;

    impl BufferMarker for TestPerFrameBuffer {
        type Size = Fixed;
        type Lifetime = PerFrameLifetime;
        const COUNT: usize = 1;
        const CONFIG: BufferConfig = BufferConfig {
            size: 16,
            usage: vk::BufferUsageFlags::empty(),
            properties: vk::MemoryPropertyFlags::empty(),
        };

        fn ids_slice() -> &'static [AtomicUsize] {
            // SAFETY: AtomicUsize and usize have identical memory layout.
            static IDS: [AtomicUsize; 1 * MAX_FRAMES_IN_FLIGHT] =
                unsafe { std::mem::transmute([usize::MAX; 1 * MAX_FRAMES_IN_FLIGHT]) };
            &IDS
        }
    }

    struct TestMutex;
    impl GpuMutex for TestMutex {}

    #[derive(Default)]
    struct TestBackend {
        upload_call: Option<(TypeId, usize, usize)>,
        per_frame_upload_call: Option<(TypeId, usize, usize)>,
    }

    impl RendererBackend for TestBackend {
        fn upload_to_buffer<
            M: GpuMutex + 'static,
            T: BufferMarker<Lifetime = Persistent> + Send + Sync + 'static,
            D: Clone + Send + Sync + 'static,
        >(
            &mut self,
            index: usize,
            data: &[D],
            dst_byte_offset: u64,
        ) {
            self.upload_call = Some((TypeId::of::<M>(), index, data.len()));
            assert_eq!(dst_byte_offset, 0);
        }

        fn upload_to_per_frame_buffer<
            M: GpuMutex + 'static,
            T: BufferMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
            D: Clone + Send + Sync + 'static,
        >(
            &mut self,
            index: usize,
            data: &[D],
        ) {
            self.per_frame_upload_call = Some((TypeId::of::<M>(), index, data.len()));
        }
    }

    #[test]
    fn upload_to_buffer_command_propagates_mutex_type_and_payload() {
        let mut backend = TestBackend::default();
        let command = UploadToBufferCommand::<TestMutex, TestBuffer, u32>::new(3, &[1, 2, 3], 0);

        command.execute(&mut backend);

        assert_eq!(backend.upload_call, Some((TypeId::of::<TestMutex>(), 3, 3)));
    }

    #[test]
    fn upload_to_per_frame_buffer_command_handles_empty_payload() {
        let mut backend = TestBackend::default();
        let command =
            UploadToPerFrameBufferCommand::<NoMutex, TestPerFrameBuffer, u32>::new(1, &[]);

        command.execute(&mut backend);

        assert_eq!(
            backend.per_frame_upload_call,
            Some((TypeId::of::<NoMutex>(), 1, 0))
        );
    }
}
