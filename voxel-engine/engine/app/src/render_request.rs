use std::marker::PhantomData;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

use crate::vulkan::resource_config::BufferMarker;
use crate::vulkan::resource_lifetime::Persistent;
use crate::vulkan::vulkan_renderer::VulkanRenderer;

pub struct GpuResponseSlot<T> {
    inner: Arc<(Mutex<Option<T>>, Condvar)>,
}

impl<T> Default for GpuResponseSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GpuResponseSlot<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(None), Condvar::new())),
        }
    }

    pub fn clone_slot(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }

    pub fn complete(&self, value: T) {
        let (lock, cvar) = &*self.inner;
        *lock.lock().unwrap() = Some(value);
        cvar.notify_one();
    }

    pub fn wait(&self) -> T {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().unwrap();
        while guard.is_none() {
            guard = cvar.wait(guard).unwrap();
        }
        guard.take().unwrap()
    }
}

pub trait RenderRequest: Send {
    fn execute(&self, renderer: &mut VulkanRenderer);
}

pub type RenderRequestSender = mpsc::Sender<Box<dyn RenderRequest>>;
pub type RenderRequestReceiver = mpsc::Receiver<Box<dyn RenderRequest>>;

pub fn create_render_request_channel() -> (RenderRequestSender, RenderRequestReceiver) {
    mpsc::channel()
}

pub struct ReadBufferDataRequest<T: BufferMarker<Lifetime = Persistent>, D: Copy, const N: usize> {
    index: usize,
    response: GpuResponseSlot<Vec<u8>>,
    _marker: PhantomData<(T, D)>,
}

impl<T: BufferMarker<Lifetime = Persistent>, D: Copy, const N: usize>
    ReadBufferDataRequest<T, D, N>
{
    pub fn new(index: usize, response: GpuResponseSlot<Vec<u8>>) -> Self {
        Self {
            index,
            response,
            _marker: PhantomData,
        }
    }
}

impl<T: BufferMarker<Lifetime = Persistent> + Send, D: Copy + Send, const N: usize> RenderRequest
    for ReadBufferDataRequest<T, D, N>
{
    fn execute(&self, renderer: &mut VulkanRenderer) {
        let data: [D; N] = renderer
            .read_buffer_data::<T, D, N>(self.index)
            .expect("Failed to read buffer data");
        let ptr = &data as *const [D; N] as *const u8;
        let bytes =
            unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<[D; N]>()).to_vec() };
        self.response.complete(bytes);
    }
}
