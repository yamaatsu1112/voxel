use std::marker::PhantomData;

use ash::vk;

use crate::rendering::GpuMutex;
use crate::rendering::renderer_command::{
    OneShot, PerFrame, RendererBackend, RendererCommand, TypedRendererCommand,
};
use crate::vulkan::resource_config::{ImageMarker, ImageViewMarker};
use crate::vulkan::resource_lifetime::{PerFrame as PerFrameLifetime, Persistent};

pub struct CreateImagesCommand<T>
where
    T: ImageMarker<Lifetime = Persistent>,
{
    pub size: Option<(u32, u32)>,
    _marker: PhantomData<T>,
}

impl<T> CreateImagesCommand<T>
where
    T: ImageMarker<Lifetime = Persistent>,
{
    pub fn new(size: Option<(u32, u32)>) -> Self {
        Self {
            size,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for CreateImagesCommand<T>
where
    T: ImageMarker<Lifetime = Persistent>,
{
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreateImagesCommand<T>
where
    T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<T, R> RendererCommand<R> for CreateImagesCommand<T>
where
    T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.create_images::<T>(self.size);
    }
}

pub struct CreateImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = Persistent>,
{
    _marker: PhantomData<T>,
}

impl<T> CreateImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = Persistent>,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for CreateImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = Persistent>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for CreateImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = Persistent>,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreateImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<T, R> RendererCommand<R> for CreateImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.create_image_views::<T>();
    }
}

pub struct CreatePerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    pub _marker: PhantomData<T>,
}

impl<T> Default for CreatePerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CreatePerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for CreatePerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreatePerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = PerFrame;
}

impl<T, R> RendererCommand<R> for CreatePerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        let frame_index = renderer.current_frame_index();
        renderer.create_per_frame_images::<T>(None, frame_index);
    }
}

pub struct CreatePerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime>,
{
    pub _marker: PhantomData<T>,
}

impl<T> Default for CreatePerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CreatePerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime>,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for CreatePerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime>,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for CreatePerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = PerFrame;
}

impl<T, R> RendererCommand<R> for CreatePerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        let frame_index = renderer.current_frame_index();
        renderer.create_per_frame_image_views::<T>(frame_index);
    }
}

pub struct DestroyPerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    pub _marker: PhantomData<T>,
}

impl<T> Default for DestroyPerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DestroyPerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for DestroyPerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for DestroyPerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = PerFrame;
}

impl<T, R> RendererCommand<R> for DestroyPerFrameImagesCommand<T>
where
    T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        let frame_index = renderer.current_frame_index();
        renderer.destroy_per_frame_images::<T>(frame_index);
    }
}

pub struct DestroyPerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime>,
{
    pub _marker: PhantomData<T>,
}

impl<T> Default for DestroyPerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DestroyPerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime>,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for DestroyPerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime>,
{
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T, R> TypedRendererCommand<R> for DestroyPerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = PerFrame;
}

impl<T, R> RendererCommand<R> for DestroyPerFrameImageViewsCommand<T>
where
    T: ImageViewMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        let frame_index = renderer.current_frame_index();
        renderer.destroy_per_frame_image_views::<T>(frame_index);
    }
}

#[allow(dead_code)]
pub struct UploadToImageCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = Persistent>,
{
    pub image_index: usize,
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    _marker: PhantomData<(M, T)>,
}

#[allow(dead_code)]
impl<M, T> UploadToImageCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = Persistent>,
{
    pub fn new(image_index: usize, data: Vec<u8>, width: u32, height: u32, depth: u32) -> Self {
        Self {
            image_index,
            data,
            width,
            height,
            depth,
            _marker: PhantomData,
        }
    }
}

impl<M, T> Clone for UploadToImageCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = Persistent>,
{
    fn clone(&self) -> Self {
        Self {
            image_index: self.image_index,
            data: self.data.clone(),
            width: self.width,
            height: self.height,
            depth: self.depth,
            _marker: PhantomData,
        }
    }
}

impl<M, T, R> TypedRendererCommand<R> for UploadToImageCommand<M, T>
where
    M: GpuMutex + Send + Sync + 'static,
    T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<M, T, R> RendererCommand<R> for UploadToImageCommand<M, T>
where
    M: GpuMutex + Send + Sync + 'static,
    T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.upload_to_image::<M, T>(
            self.image_index,
            &self.data,
            self.width,
            self.height,
            self.depth,
        );
    }
}

#[allow(dead_code)]
pub struct TransitionLayoutCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = Persistent>,
{
    pub index: usize,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub aspect_mask: vk::ImageAspectFlags,
    _marker: PhantomData<(M, T)>,
}

#[allow(dead_code)]
impl<M, T> TransitionLayoutCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = Persistent>,
{
    pub fn new(
        index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Self {
        Self {
            index,
            old_layout,
            new_layout,
            aspect_mask,
            _marker: PhantomData,
        }
    }
}

impl<M, T> Clone for TransitionLayoutCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = Persistent>,
{
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            old_layout: self.old_layout,
            new_layout: self.new_layout,
            aspect_mask: self.aspect_mask,
            _marker: PhantomData,
        }
    }
}

impl<M, T, R> TypedRendererCommand<R> for TransitionLayoutCommand<M, T>
where
    M: GpuMutex + Send + Sync + 'static,
    T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = OneShot;
}

impl<M, T, R> RendererCommand<R> for TransitionLayoutCommand<M, T>
where
    M: GpuMutex + Send + Sync + 'static,
    T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        renderer.transition_image_layout::<M, T>(
            self.index,
            self.old_layout,
            self.new_layout,
            self.aspect_mask,
        );
    }
}

pub struct TransitionPerFrameLayoutCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    pub image_index: usize,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub aspect_mask: vk::ImageAspectFlags,
    _marker: PhantomData<(M, T)>,
}

impl<M, T> TransitionPerFrameLayoutCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    pub fn new(
        image_index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Self {
        Self {
            image_index,
            old_layout,
            new_layout,
            aspect_mask,
            _marker: PhantomData,
        }
    }
}

impl<M, T> Clone for TransitionPerFrameLayoutCommand<M, T>
where
    M: GpuMutex,
    T: ImageMarker<Lifetime = PerFrameLifetime>,
{
    fn clone(&self) -> Self {
        Self {
            image_index: self.image_index,
            old_layout: self.old_layout,
            new_layout: self.new_layout,
            aspect_mask: self.aspect_mask,
            _marker: PhantomData,
        }
    }
}

impl<M, T, R> TypedRendererCommand<R> for TransitionPerFrameLayoutCommand<M, T>
where
    M: GpuMutex + Send + Sync + 'static,
    T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    type CmdType = PerFrame;
}

impl<M, T, R> RendererCommand<R> for TransitionPerFrameLayoutCommand<M, T>
where
    M: GpuMutex + Send + Sync + 'static,
    T: ImageMarker<Lifetime = PerFrameLifetime> + Send + Sync + 'static,
    R: RendererBackend,
{
    fn clone_box(&self) -> Box<dyn RendererCommand<R>> {
        Box::new((*self).clone())
    }

    fn execute(&self, renderer: &mut R) {
        let frame_index = renderer.current_frame_index();
        renderer.transition_per_frame_image_layout::<M, T>(
            frame_index,
            self.image_index,
            self.old_layout,
            self.new_layout,
            self.aspect_mask,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{TransitionLayoutCommand, UploadToImageCommand};
    use crate::rendering::renderer_command::{RendererBackend, RendererCommand};
    use crate::rendering::{GpuMutex, NoMutex};
    use crate::vulkan::resource_config::{ImageConfig, ImageMarker, ImageSizeFormat};
    use crate::vulkan::resource_lifetime::Persistent;
    use ash::vk;
    use std::any::TypeId;

    engine_macro::define_image! {
        struct TestImage;
        lifetime = Persistent;
        count = 1;
        config = ImageConfig {
            size: ImageSizeFormat::Fixed {
                width: 1,
                height: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::empty(),
            properties: vk::MemoryPropertyFlags::empty(),
        };
    }

    struct TestMutex;
    impl GpuMutex for TestMutex {}

    #[derive(Default)]
    struct TestBackend {
        current_frame: usize,
        upload_call: Option<(TypeId, usize, usize, u32, u32, u32)>,
        transition_call: Option<(
            TypeId,
            usize,
            vk::ImageLayout,
            vk::ImageLayout,
            vk::ImageAspectFlags,
        )>,
    }

    impl RendererBackend for TestBackend {
        fn current_frame_index(&self) -> usize {
            self.current_frame
        }

        fn upload_to_image<
            M: GpuMutex + 'static,
            T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
        >(
            &mut self,
            image_index: usize,
            data: &[u8],
            width: u32,
            height: u32,
            depth: u32,
        ) {
            self.upload_call = Some((
                TypeId::of::<M>(),
                image_index,
                data.len(),
                width,
                height,
                depth,
            ));
        }

        fn transition_image_layout<
            M: GpuMutex + 'static,
            T: ImageMarker<Lifetime = Persistent> + Send + Sync + 'static,
        >(
            &mut self,
            index: usize,
            old_layout: vk::ImageLayout,
            new_layout: vk::ImageLayout,
            aspect_mask: vk::ImageAspectFlags,
        ) {
            self.transition_call = Some((
                TypeId::of::<M>(),
                index,
                old_layout,
                new_layout,
                aspect_mask,
            ));
        }
    }

    #[test]
    fn upload_to_image_command_propagates_mutex_and_dimensions() {
        let mut backend = TestBackend::default();
        let command =
            UploadToImageCommand::<TestMutex, TestImage>::new(2, vec![1, 2, 3, 4], 1, 1, 1);

        command.execute(&mut backend);

        assert_eq!(
            backend.upload_call,
            Some((TypeId::of::<TestMutex>(), 2, 4, 1, 1, 1))
        );
    }

    #[test]
    fn upload_to_image_command_propagates_depth() {
        let mut backend = TestBackend::default();
        let data = vec![0u8; 2 * 2 * 4 * 4]; // 2x2x4 RGBA
        let command = UploadToImageCommand::<TestMutex, TestImage>::new(0, data.clone(), 2, 2, 4);

        command.execute(&mut backend);

        assert_eq!(
            backend.upload_call,
            Some((TypeId::of::<TestMutex>(), 0, data.len(), 2, 2, 4))
        );
    }

    #[test]
    fn transition_layout_command_propagates_noop_mutex() {
        let mut backend = TestBackend::default();
        let command = TransitionLayoutCommand::<NoMutex, TestImage>::new(
            0,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::ImageAspectFlags::COLOR,
        );

        command.execute(&mut backend);

        assert_eq!(
            backend.transition_call,
            Some((
                TypeId::of::<NoMutex>(),
                0,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
            ))
        );
    }
}
