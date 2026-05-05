use ash::vk;
use std::ops::BitOr;
use std::sync::atomic::Ordering;

use crate::vulkan::command_resource_manager::CommandResourceManager;
use crate::vulkan::descriptors::DescriptorManager;
use crate::vulkan::pipeline_manager::PipelineManager;
use crate::vulkan::record_resource::RecordResource;
use crate::vulkan::resource_config::{
    BufferMarker, ComputePipelineMarker, DescriptorSetId, DescriptorSetMarker,
    GraphicsPipelineMarker, ImageMarker, ImageViewMarker, ReusableCommandBufferMarker, buffer_id,
    compute_pipeline_id, descriptor_set_id, graphics_pipeline_id, image_id, image_view_id,
    per_frame_buffer_id, per_frame_descriptor_set_id, per_frame_image_id, per_frame_image_view_id,
    reusable_command_buffer_slot,
};
use crate::vulkan::resource_lifetime::{PerFrame, Persistent, ResourceLifetime};
use crate::vulkan::resource_manager::ResourceManager;
use crate::vulkan::resources::command_resources::{
    ComputeCommandResources, RendererCommandResources,
};
use crate::vulkan::swapchain::SwapchainManager;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderStageFlags(u32);

impl ShaderStageFlags {
    pub const VERTEX: Self = Self(0x1);
    pub const FRAGMENT: Self = Self(0x10);
    pub const COMPUTE: Self = Self(0x20);

    fn to_vk(self) -> vk::ShaderStageFlags {
        let mut flags = vk::ShaderStageFlags::empty();
        if self.0 & Self::VERTEX.0 != 0 {
            flags |= vk::ShaderStageFlags::VERTEX;
        }
        if self.0 & Self::FRAGMENT.0 != 0 {
            flags |= vk::ShaderStageFlags::FRAGMENT;
        }
        if self.0 & Self::COMPUTE.0 != 0 {
            flags |= vk::ShaderStageFlags::COMPUTE;
        }
        flags
    }
}

impl BitOr for ShaderStageFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexType {
    Uint16,
    Uint32,
}

impl IndexType {
    fn to_vk(self) -> vk::IndexType {
        match self {
            Self::Uint16 => vk::IndexType::UINT16,
            Self::Uint32 => vk::IndexType::UINT32,
        }
    }
}

#[derive(Clone, Copy)]
pub struct SecondaryCommandBufferHandle(pub(crate) vk::CommandBuffer);

pub trait BufferResolver: ResourceLifetime {
    fn get_buffer_id<B: BufferMarker<Lifetime = Self>>(frame: usize, index: usize) -> Option<usize>
    where
        Self: Sized;
}

impl BufferResolver for Persistent {
    fn get_buffer_id<B: BufferMarker<Lifetime = Self>>(
        _frame: usize,
        index: usize,
    ) -> Option<usize> {
        buffer_id::<B>(index).map(|id| id.0)
    }
}

impl BufferResolver for PerFrame {
    fn get_buffer_id<B: BufferMarker<Lifetime = Self>>(
        frame: usize,
        index: usize,
    ) -> Option<usize> {
        per_frame_buffer_id::<B>(frame, index).map(|id| id.0)
    }
}

pub trait DescriptorSetResolver: ResourceLifetime {
    fn get_descriptor_set_id<D: DescriptorSetMarker<Lifetime = Self>>(
        frame: usize,
    ) -> Option<DescriptorSetId>
    where
        Self: Sized;
}

impl DescriptorSetResolver for Persistent {
    fn get_descriptor_set_id<D: DescriptorSetMarker<Lifetime = Self>>(
        _frame: usize,
    ) -> Option<DescriptorSetId> {
        descriptor_set_id::<D>()
    }
}

impl DescriptorSetResolver for PerFrame {
    fn get_descriptor_set_id<D: DescriptorSetMarker<Lifetime = Self>>(
        frame: usize,
    ) -> Option<DescriptorSetId> {
        per_frame_descriptor_set_id::<D>(frame)
    }
}

pub trait ImageResolver: ResourceLifetime {
    fn get_image_id<I: ImageMarker<Lifetime = Self>>(frame: usize, index: usize) -> Option<usize>
    where
        Self: Sized;
}

impl ImageResolver for Persistent {
    fn get_image_id<I: ImageMarker<Lifetime = Self>>(_frame: usize, index: usize) -> Option<usize> {
        image_id::<I>(index).map(|id| id.0)
    }
}

impl ImageResolver for PerFrame {
    fn get_image_id<I: ImageMarker<Lifetime = Self>>(frame: usize, index: usize) -> Option<usize> {
        per_frame_image_id::<I>(frame, index).map(|id| id.0)
    }
}

pub trait ImageViewResolver: ResourceLifetime {
    fn get_image_view_id<IV: ImageViewMarker<Lifetime = Self>>(
        frame: usize,
        index: usize,
    ) -> Option<usize>
    where
        Self: Sized;
}

impl ImageViewResolver for Persistent {
    fn get_image_view_id<IV: ImageViewMarker<Lifetime = Self>>(
        _frame: usize,
        index: usize,
    ) -> Option<usize> {
        image_view_id::<IV>(index).map(|id| id.0)
    }
}

impl ImageViewResolver for PerFrame {
    fn get_image_view_id<IV: ImageViewMarker<Lifetime = Self>>(
        frame: usize,
        index: usize,
    ) -> Option<usize> {
        per_frame_image_view_id::<IV>(frame, index).map(|id| id.0)
    }
}

fn descriptor_set_or_none(
    descriptor_manager: &DescriptorManager,
    descriptor_set_id: Option<DescriptorSetId>,
) -> Option<vk::DescriptorSet> {
    let id = descriptor_set_id?;
    if id == DescriptorSetId::INVALID {
        return None;
    }
    descriptor_manager.get_descriptor_set(id)
}

fn push_constant_bytes<T>(data: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(data).cast::<u8>(),
            std::mem::size_of::<T>(),
        )
    }
}

pub struct VkGraphicsRecordContext<'a> {
    command_buffer: vk::CommandBuffer,
    current_frame: usize,
    image_index: usize,
    device: &'a ash::Device,
    pipeline_manager: &'a PipelineManager,
    descriptor_manager: &'a DescriptorManager,
    resource_manager: &'a ResourceManager,
    _command_resource_manager: &'a mut CommandResourceManager,
    _command_resources: &'a RendererCommandResources,
    swapchain: &'a SwapchainManager,
    record_resource: &'a RecordResource,
    #[cfg(feature = "gpu-profiling")]
    gpu_timing: &'a mut crate::vulkan::gpu_timing::GpuTimingManager,
}

impl<'a> VkGraphicsRecordContext<'a> {
    pub(crate) fn new(
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
        current_frame: usize,
        image_index: usize,
        pipeline_manager: &'a PipelineManager,
        descriptor_manager: &'a DescriptorManager,
        resource_manager: &'a ResourceManager,
        command_resource_manager: &'a mut CommandResourceManager,
        command_resources: &'a RendererCommandResources,
        swapchain: &'a SwapchainManager,
        record_resource: &'a RecordResource,
        #[cfg(feature = "gpu-profiling")]
        gpu_timing: &'a mut crate::vulkan::gpu_timing::GpuTimingManager,
    ) -> Self {
        Self {
            command_buffer,
            current_frame,
            image_index,
            device,
            pipeline_manager,
            descriptor_manager,
            resource_manager,
            _command_resource_manager: command_resource_manager,
            _command_resources: command_resources,
            swapchain,
            record_resource,
            #[cfg(feature = "gpu-profiling")]
            gpu_timing,
        }
    }

    pub fn cmd_begin_timing(&mut self, label: &str) {
        #[cfg(feature = "gpu-profiling")]
        {
            self.gpu_timing.cmd_begin_timing(
                self.device,
                self.command_buffer,
                self.current_frame,
                label,
            );
        }
        let _ = label;
    }

    pub fn cmd_end_timing(&mut self, label: &str) {
        #[cfg(feature = "gpu-profiling")]
        {
            self.gpu_timing.cmd_end_timing(
                self.device,
                self.command_buffer,
                self.current_frame,
                label,
            );
        }
        let _ = label;
    }

    pub fn cmd_bind_graphics_pipeline<P: GraphicsPipelineMarker>(&self) {
        let id = graphics_pipeline_id::<P>();
        let (pipeline, _) = self
            .pipeline_manager
            .get_graphics_pipeline(id)
            .expect("Graphics pipeline not found");
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
        }
    }

    pub fn cmd_bind_descriptor_set<P: GraphicsPipelineMarker, D: DescriptorSetMarker>(
        &self,
        first_set: u32,
        dynamic_offsets: &[u32],
    ) where
        D::Lifetime: DescriptorSetResolver,
    {
        let descriptor_set = descriptor_set_or_none(
            self.descriptor_manager,
            <D::Lifetime as DescriptorSetResolver>::get_descriptor_set_id::<D>(self.current_frame),
        )
        .expect("Descriptor set not found");
        let (_, layout) = self
            .pipeline_manager
            .get_graphics_pipeline(graphics_pipeline_id::<P>())
            .expect("Graphics pipeline not found");

        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                first_set,
                &[descriptor_set],
                dynamic_offsets,
            );
        }
    }

    pub fn cmd_bind_vertex_buffer<B: BufferMarker>(&self, binding: u32, index: usize, offset: u64)
    where
        B::Lifetime: BufferResolver,
    {
        let buffer_id =
            <B::Lifetime as BufferResolver>::get_buffer_id::<B>(self.current_frame, index)
                .expect("Buffer id not found");
        let buffer = self
            .resource_manager
            .get_buffer(crate::vulkan::resource_config::BufferId(buffer_id))
            .expect("Buffer not found");

        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.command_buffer, binding, &[buffer], &[offset]);
        }
    }

    pub fn cmd_bind_index_buffer<B: BufferMarker>(
        &self,
        index: usize,
        offset: u64,
        index_type: IndexType,
    ) where
        B::Lifetime: BufferResolver,
    {
        let buffer_id =
            <B::Lifetime as BufferResolver>::get_buffer_id::<B>(self.current_frame, index)
                .expect("Buffer id not found");
        let buffer = self
            .resource_manager
            .get_buffer(crate::vulkan::resource_config::BufferId(buffer_id))
            .expect("Buffer not found");

        unsafe {
            self.device.cmd_bind_index_buffer(
                self.command_buffer,
                buffer,
                offset,
                index_type.to_vk(),
            );
        }
    }

    pub fn cmd_draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw(
                self.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub fn cmd_draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    pub fn cmd_push_constants<P: GraphicsPipelineMarker, T>(
        &self,
        stage_flags: ShaderStageFlags,
        offset: u32,
        data: &T,
    ) {
        let (_, layout) = self
            .pipeline_manager
            .get_graphics_pipeline(graphics_pipeline_id::<P>())
            .expect("Graphics pipeline not found");
        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                layout,
                stage_flags.to_vk(),
                offset,
                push_constant_bytes(data),
            );
        }
    }

    pub fn cmd_set_viewport(
        &self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        let viewport = vk::Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        };
        unsafe {
            self.device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport]);
        }
    }

    pub fn cmd_set_scissor(&self, offset_x: i32, offset_y: i32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: offset_x,
                y: offset_y,
            },
            extent: vk::Extent2D { width, height },
        };
        unsafe {
            self.device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        }
    }

    pub fn cmd_begin_rendering(&self, rendering_info: &vk::RenderingInfo) {
        unsafe {
            self.device
                .cmd_begin_rendering(self.command_buffer, rendering_info);
        }
    }

    pub fn cmd_end_rendering(&self) {
        unsafe {
            self.device.cmd_end_rendering(self.command_buffer);
        }
    }

    pub fn cmd_copy_buffer<Src: BufferMarker, Dst: BufferMarker>(
        &self,
        src_index: usize,
        dst_index: usize,
        regions: &[vk::BufferCopy],
    ) where
        Src::Lifetime: BufferResolver,
        Dst::Lifetime: BufferResolver,
    {
        let src = self
            .resource_manager
            .get_buffer(crate::vulkan::resource_config::BufferId(
                <Src::Lifetime as BufferResolver>::get_buffer_id::<Src>(
                    self.current_frame,
                    src_index,
                )
                .expect("Source buffer id not found"),
            ))
            .expect("Source buffer not found");
        let dst = self
            .resource_manager
            .get_buffer(crate::vulkan::resource_config::BufferId(
                <Dst::Lifetime as BufferResolver>::get_buffer_id::<Dst>(
                    self.current_frame,
                    dst_index,
                )
                .expect("Destination buffer id not found"),
            ))
            .expect("Destination buffer not found");

        unsafe {
            self.device
                .cmd_copy_buffer(self.command_buffer, src, dst, regions);
        }
    }

    pub fn cmd_pipeline_barrier(
        &self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_barriers: &[vk::BufferMemoryBarrier],
        image_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                dependency_flags,
                memory_barriers,
                buffer_barriers,
                image_barriers,
            );
        }
    }

    pub fn get_buffer_size<B: BufferMarker>(&self, index: usize) -> Option<u64>
    where
        B::Lifetime: BufferResolver,
    {
        let buffer_id =
            <B::Lifetime as BufferResolver>::get_buffer_id::<B>(self.current_frame, index)?;
        self.resource_manager
            .get_buffer_size(crate::vulkan::resource_config::BufferId(buffer_id))
    }

    pub fn get_image_extent<I: ImageMarker>(&self, index: usize) -> Option<vk::Extent3D>
    where
        I::Lifetime: ImageResolver,
    {
        let image_id =
            <I::Lifetime as ImageResolver>::get_image_id::<I>(self.current_frame, index)?;
        self.resource_manager
            .get_image_extent(crate::vulkan::resource_config::ImageId(image_id))
    }

    pub fn get_swapchain_extent(&self) -> (u32, u32) {
        let extent = self.swapchain.get_extent();
        (extent.width, extent.height)
    }

    pub fn get_swapchain_image(&self, image_index: usize) -> vk::Image {
        self.swapchain.get_images()[image_index]
    }

    pub fn get_swapchain_image_view(&self, image_index: usize) -> vk::ImageView {
        self.swapchain.get_image_views()[image_index]
    }

    pub fn get_per_frame_image<I: ImageMarker>(&self, index: usize) -> Option<vk::Image>
    where
        I::Lifetime: ImageResolver,
    {
        let image_id =
            <I::Lifetime as ImageResolver>::get_image_id::<I>(self.current_frame, index)?;
        self.resource_manager
            .get_image(crate::vulkan::resource_config::ImageId(image_id))
    }

    pub fn get_per_frame_image_view<IV: ImageViewMarker>(
        &self,
        index: usize,
    ) -> Option<vk::ImageView>
    where
        IV::Lifetime: ImageViewResolver,
    {
        let image_view_id = <IV::Lifetime as ImageViewResolver>::get_image_view_id::<IV>(
            self.current_frame,
            index,
        )?;
        self.resource_manager
            .get_image_view(crate::vulkan::resource_config::ImageViewId(image_view_id))
    }

    pub fn record_resource(&self) -> &RecordResource {
        self.record_resource
    }

    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    pub fn image_index(&self) -> usize {
        self.image_index
    }
}

pub struct VkComputeRecordContext<'a> {
    command_buffer: vk::CommandBuffer,
    current_compute_frame: usize,
    device: &'a ash::Device,
    pipeline_manager: &'a PipelineManager,
    descriptor_manager: &'a DescriptorManager,
    resource_manager: &'a ResourceManager,
    command_resource_manager: &'a mut CommandResourceManager,
    command_resources: &'a ComputeCommandResources,
    record_resource: &'a RecordResource,
    #[cfg(feature = "gpu-profiling")]
    gpu_timing: &'a mut crate::vulkan::gpu_timing::GpuTimingManager,
}

impl<'a> VkComputeRecordContext<'a> {
    pub(crate) fn new(
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
        current_compute_frame: usize,
        pipeline_manager: &'a PipelineManager,
        descriptor_manager: &'a DescriptorManager,
        resource_manager: &'a ResourceManager,
        command_resource_manager: &'a mut CommandResourceManager,
        command_resources: &'a ComputeCommandResources,
        record_resource: &'a RecordResource,
        #[cfg(feature = "gpu-profiling")]
        gpu_timing: &'a mut crate::vulkan::gpu_timing::GpuTimingManager,
    ) -> Self {
        Self {
            command_buffer,
            current_compute_frame,
            device,
            pipeline_manager,
            descriptor_manager,
            resource_manager,
            command_resource_manager,
            command_resources,
            record_resource,
            #[cfg(feature = "gpu-profiling")]
            gpu_timing,
        }
    }

    pub fn cmd_begin_timing(&mut self, label: &str) {
        #[cfg(feature = "gpu-profiling")]
        {
            self.gpu_timing.cmd_begin_timing(
                self.device,
                self.command_buffer,
                self.current_compute_frame,
                label,
            );
        }
        let _ = label;
    }

    pub fn cmd_end_timing(&mut self, label: &str) {
        #[cfg(feature = "gpu-profiling")]
        {
            self.gpu_timing.cmd_end_timing(
                self.device,
                self.command_buffer,
                self.current_compute_frame,
                label,
            );
        }
        let _ = label;
    }

    pub fn cmd_bind_compute_pipeline<P: ComputePipelineMarker>(&self) {
        let (pipeline, _) = self
            .pipeline_manager
            .get_compute_pipeline(compute_pipeline_id::<P>())
            .expect("Compute pipeline not found");
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
        }
    }

    pub fn cmd_bind_descriptor_set<P: ComputePipelineMarker, D: DescriptorSetMarker>(
        &self,
        first_set: u32,
        dynamic_offsets: &[u32],
    ) where
        D::Lifetime: DescriptorSetResolver,
    {
        let descriptor_set = descriptor_set_or_none(
            self.descriptor_manager,
            <D::Lifetime as DescriptorSetResolver>::get_descriptor_set_id::<D>(
                self.current_compute_frame,
            ),
        )
        .expect("Descriptor set not found");
        let (_, layout) = self
            .pipeline_manager
            .get_compute_pipeline(compute_pipeline_id::<P>())
            .expect("Compute pipeline not found");

        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                first_set,
                &[descriptor_set],
                dynamic_offsets,
            );
        }
    }

    pub fn cmd_push_constants<P: ComputePipelineMarker, T>(
        &self,
        stage_flags: ShaderStageFlags,
        offset: u32,
        data: &T,
    ) {
        let (_, layout) = self
            .pipeline_manager
            .get_compute_pipeline(compute_pipeline_id::<P>())
            .expect("Compute pipeline not found");

        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                layout,
                stage_flags.to_vk(),
                offset,
                push_constant_bytes(data),
            );
        }
    }

    pub fn cmd_dispatch(&self, x: u32, y: u32, z: u32) {
        unsafe {
            self.device.cmd_dispatch(self.command_buffer, x, y, z);
        }
    }

    pub fn cmd_dispatch_indirect<B: BufferMarker>(&self, index: usize, offset: u64)
    where
        B::Lifetime: BufferResolver,
    {
        let buffer = self
            .resource_manager
            .get_buffer(crate::vulkan::resource_config::BufferId(
                <B::Lifetime as BufferResolver>::get_buffer_id::<B>(
                    self.current_compute_frame,
                    index,
                )
                .expect("Indirect buffer id not found"),
            ))
            .expect("Indirect buffer not found");

        unsafe {
            self.device
                .cmd_dispatch_indirect(self.command_buffer, buffer, offset);
        }
    }

    pub fn cmd_pipeline_barrier(
        &self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_barriers: &[vk::BufferMemoryBarrier],
        image_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                dependency_flags,
                memory_barriers,
                buffer_barriers,
                image_barriers,
            );
        }
    }

    pub fn cmd_execute_commands(&self, secondaries: &[SecondaryCommandBufferHandle]) {
        let buffers: Vec<_> = secondaries.iter().map(|handle| handle.0).collect();
        unsafe {
            self.device
                .cmd_execute_commands(self.command_buffer, &buffers);
        }
    }

    pub fn get_reusable_secondary<M: ReusableCommandBufferMarker>(
        &mut self,
        index: usize,
    ) -> (VkSecondaryRecordContext<'_>, bool) {
        let reusable_pool_id =
            self.command_resources.reusable_compute_pools[self.current_compute_frame];
        let slot = reusable_command_buffer_slot::<M>(self.current_compute_frame, index);
        let buffer_index = slot.load(Ordering::Acquire);

        let (command_buffer, is_new) = if buffer_index == usize::MAX {
            let (command_buffer, buffer_index) = self
                .command_resource_manager
                .get_free_command_buffer_with_index(
                    reusable_pool_id,
                    vk::CommandBufferLevel::SECONDARY,
                );
            slot.store(buffer_index, Ordering::Release);
            (command_buffer, true)
        } else {
            let command_buffer = self
                .command_resource_manager
                .get_command_buffer_in_pool(
                    reusable_pool_id,
                    vk::CommandBufferLevel::SECONDARY,
                    buffer_index,
                )
                .expect("Reusable command buffer index was not allocated");
            (command_buffer, false)
        };

        (
            VkSecondaryRecordContext::new(
                self.device,
                command_buffer,
                self.pipeline_manager,
                self.descriptor_manager,
                self.resource_manager,
                self.current_compute_frame,
            ),
            is_new,
        )
    }

    pub fn get_free_secondary(&mut self) -> VkSecondaryRecordContext<'_> {
        let pool_id =
            self.command_resources.compute_secondary_command_pools[self.current_compute_frame];
        let command_buffer = self
            .command_resource_manager
            .get_free_command_buffer(pool_id, vk::CommandBufferLevel::SECONDARY);

        VkSecondaryRecordContext::new(
            self.device,
            command_buffer,
            self.pipeline_manager,
            self.descriptor_manager,
            self.resource_manager,
            self.current_compute_frame,
        )
    }

    pub fn get_buffer_size<B: BufferMarker>(&self, index: usize) -> Option<u64>
    where
        B::Lifetime: BufferResolver,
    {
        let buffer_id =
            <B::Lifetime as BufferResolver>::get_buffer_id::<B>(self.current_compute_frame, index)?;
        self.resource_manager
            .get_buffer_size(crate::vulkan::resource_config::BufferId(buffer_id))
    }

    pub fn record_resource(&self) -> &RecordResource {
        self.record_resource
    }

    pub fn current_compute_frame(&self) -> usize {
        self.current_compute_frame
    }
}

pub struct VkSecondaryRecordContext<'a> {
    device: &'a ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline_manager: &'a PipelineManager,
    descriptor_manager: &'a DescriptorManager,
    resource_manager: &'a ResourceManager,
    current_compute_frame: usize,
}

impl<'a> VkSecondaryRecordContext<'a> {
    fn new(
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
        pipeline_manager: &'a PipelineManager,
        descriptor_manager: &'a DescriptorManager,
        resource_manager: &'a ResourceManager,
        current_compute_frame: usize,
    ) -> Self {
        Self {
            device,
            command_buffer,
            pipeline_manager,
            descriptor_manager,
            resource_manager,
            current_compute_frame,
        }
    }

    pub fn begin(&self) {
        self.begin_with_flags(vk::CommandBufferUsageFlags::empty());
    }

    pub fn begin_with_flags(&self, flags: vk::CommandBufferUsageFlags) {
        let inheritance_info = vk::CommandBufferInheritanceInfo::default();
        let begin_info = vk::CommandBufferBeginInfo {
            flags,
            p_inheritance_info: &inheritance_info,
            ..Default::default()
        };

        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .expect("Failed to begin secondary command buffer");
        }
    }

    pub fn end(&self) {
        unsafe {
            self.device
                .end_command_buffer(self.command_buffer)
                .expect("Failed to end secondary command buffer");
        }
    }

    pub fn handle(&self) -> SecondaryCommandBufferHandle {
        SecondaryCommandBufferHandle(self.command_buffer)
    }

    pub fn cmd_bind_compute_pipeline<P: ComputePipelineMarker>(&self) {
        let (pipeline, _) = self
            .pipeline_manager
            .get_compute_pipeline(compute_pipeline_id::<P>())
            .expect("Compute pipeline not found");
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
        }
    }

    pub fn cmd_bind_descriptor_set<P: ComputePipelineMarker, D: DescriptorSetMarker>(
        &self,
        first_set: u32,
        dynamic_offsets: &[u32],
    ) where
        D::Lifetime: DescriptorSetResolver,
    {
        let descriptor_set = descriptor_set_or_none(
            self.descriptor_manager,
            <D::Lifetime as DescriptorSetResolver>::get_descriptor_set_id::<D>(
                self.current_compute_frame,
            ),
        )
        .expect("Descriptor set not found");
        let (_, layout) = self
            .pipeline_manager
            .get_compute_pipeline(compute_pipeline_id::<P>())
            .expect("Compute pipeline not found");

        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                first_set,
                &[descriptor_set],
                dynamic_offsets,
            );
        }
    }

    pub fn cmd_push_constants<P: ComputePipelineMarker, T>(
        &self,
        stage_flags: ShaderStageFlags,
        offset: u32,
        data: &T,
    ) {
        let (_, layout) = self
            .pipeline_manager
            .get_compute_pipeline(compute_pipeline_id::<P>())
            .expect("Compute pipeline not found");

        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                layout,
                stage_flags.to_vk(),
                offset,
                push_constant_bytes(data),
            );
        }
    }

    pub fn cmd_dispatch(&self, x: u32, y: u32, z: u32) {
        unsafe {
            self.device.cmd_dispatch(self.command_buffer, x, y, z);
        }
    }

    pub fn cmd_dispatch_indirect<B: BufferMarker>(&self, index: usize, offset: u64)
    where
        B::Lifetime: BufferResolver,
    {
        let buffer = self
            .resource_manager
            .get_buffer(crate::vulkan::resource_config::BufferId(
                <B::Lifetime as BufferResolver>::get_buffer_id::<B>(
                    self.current_compute_frame,
                    index,
                )
                .expect("Indirect buffer id not found"),
            ))
            .expect("Indirect buffer not found");

        unsafe {
            self.device
                .cmd_dispatch_indirect(self.command_buffer, buffer, offset);
        }
    }

    pub fn cmd_pipeline_barrier(
        &self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_barriers: &[vk::BufferMemoryBarrier],
        image_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                dependency_flags,
                memory_barriers,
                buffer_barriers,
                image_barriers,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IndexType, ShaderStageFlags};
    use ash::vk;

    #[test]
    fn shader_stage_flags_maps_to_vk_flags() {
        let flags = (ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT).to_vk();
        assert!(flags.contains(vk::ShaderStageFlags::VERTEX));
        assert!(flags.contains(vk::ShaderStageFlags::FRAGMENT));
        assert!(!flags.contains(vk::ShaderStageFlags::COMPUTE));
    }

    #[test]
    fn index_type_maps_to_vk_index_type() {
        assert_eq!(IndexType::Uint16.to_vk(), vk::IndexType::UINT16);
        assert_eq!(IndexType::Uint32.to_vk(), vk::IndexType::UINT32);
    }
}
