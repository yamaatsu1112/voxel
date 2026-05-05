use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};

use ash::vk;

use crate::rendering::gpu_mutex::{GpuMutex, GpuMutexList};
use crate::vulkan::command_resource_manager::CommandResourceManager;
use crate::vulkan::descriptors::DescriptorManager;
use crate::vulkan::gpu_mutex::GpuMutexState;
use crate::vulkan::pipeline_manager::PipelineManager;
use crate::vulkan::record_context::VkComputeRecordContext;
use crate::vulkan::record_resource::RecordResource;
use crate::vulkan::recordable::VkComputeRecordable;
use crate::vulkan::resource_config::BufferMarker;
use crate::vulkan::resource_lifetime::Persistent;
use crate::vulkan::resource_manager::ResourceManager;
use crate::vulkan::resources::ComputeCommandResources;
use crate::vulkan::staging_ring_buffer::StagingRingBuffer;
use crate::vulkan::submit_group::ComputeSubmitGroup;
use crate::vulkan::sync::create_timeline_semaphore;
use crate::vulkan::vulkan_renderer_core::{
    MAX_COMPUTE_FRAMES_IN_FLIGHT, acquire_mutex_state, begin_transfer_command_buffer,
    submit_transfer_command_buffer, wait_for_timeline_value,
};

pub struct ComputeData {
    current_compute_frame: usize,
    compute_record_resource: RecordResource,
    compute_fences: HashMap<TypeId, [vk::Fence; MAX_COMPUTE_FRAMES_IN_FLIGHT]>,
    compute_staging_buffers: [StagingRingBuffer; MAX_COMPUTE_FRAMES_IN_FLIGHT],
}

impl ComputeData {
    pub(crate) fn new(
        compute_staging_buffers: [StagingRingBuffer; MAX_COMPUTE_FRAMES_IN_FLIGHT],
    ) -> Self {
        Self {
            current_compute_frame: 0,
            compute_record_resource: RecordResource::new(),
            compute_fences: HashMap::new(),
            compute_staging_buffers,
        }
    }
}

pub struct ComputeExecutor {
    device: ash::Device,
    compute_command_resource_manager: CommandResourceManager,
    compute_command_resources: ComputeCommandResources,
    compute_data: ComputeData,
    compute_queue: vk::Queue,
    compute_transfer_queue: vk::Queue,
    pipeline_manager: Arc<RwLock<PipelineManager>>,
    descriptor_manager: Arc<RwLock<DescriptorManager>>,
    resource_manager: Arc<RwLock<ResourceManager>>,
    gpu_mutexes: Arc<Mutex<HashMap<TypeId, GpuMutexState>>>,
    physical_device: vk::PhysicalDevice,
    instance: ash::Instance,
    queue_family_indices: Vec<u32>,
    compute_transfer_timeline: vk::Semaphore,
    compute_transfer_next_timeline_value: u64,
    compute_transfer_last_submitted_values: [u64; MAX_COMPUTE_FRAMES_IN_FLIGHT],
    #[cfg(feature = "gpu-profiling")]
    gpu_timing: crate::vulkan::gpu_timing::GpuTimingManager,
}

impl ComputeExecutor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: ash::Device,
        compute_command_resource_manager: CommandResourceManager,
        compute_command_resources: ComputeCommandResources,
        compute_data: ComputeData,
        compute_queue: vk::Queue,
        compute_transfer_queue: vk::Queue,
        pipeline_manager: Arc<RwLock<PipelineManager>>,
        descriptor_manager: Arc<RwLock<DescriptorManager>>,
        resource_manager: Arc<RwLock<ResourceManager>>,
        gpu_mutexes: Arc<Mutex<HashMap<TypeId, GpuMutexState>>>,
        physical_device: vk::PhysicalDevice,
        instance: ash::Instance,
        queue_family_indices: Vec<u32>,
    ) -> Self {
        let compute_transfer_timeline = create_timeline_semaphore(&device);
        #[cfg(feature = "gpu-profiling")]
        let gpu_timing = {
            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            crate::vulkan::gpu_timing::GpuTimingManager::new(&device, props.limits.timestamp_period)
        };
        Self {
            device,
            compute_command_resource_manager,
            compute_command_resources,
            compute_data,
            compute_queue,
            compute_transfer_queue,
            pipeline_manager,
            descriptor_manager,
            resource_manager,
            gpu_mutexes,
            physical_device,
            instance,
            queue_family_indices,
            compute_transfer_timeline,
            compute_transfer_next_timeline_value: 1,
            compute_transfer_last_submitted_values: [0; MAX_COMPUTE_FRAMES_IN_FLIGHT],
            #[cfg(feature = "gpu-profiling")]
            gpu_timing,
        }
    }

    pub fn begin_compute(&mut self) {
        self.compute_data.current_compute_frame =
            (self.compute_data.current_compute_frame + 1) % MAX_COMPUTE_FRAMES_IN_FLIGHT;

        let fences: Vec<vk::Fence> = self
            .compute_data
            .compute_fences
            .values()
            .map(|f| f[self.compute_data.current_compute_frame])
            .collect();
        if !fences.is_empty() {
            unsafe {
                let _ = self.device.wait_for_fences(&fences, true, u64::MAX);
            }
        }
        wait_for_timeline_value(
            &self.device,
            self.compute_transfer_timeline,
            self.compute_transfer_last_submitted_values[self.compute_data.current_compute_frame],
        );

        #[cfg(feature = "gpu-profiling")]
        {
            self.gpu_timing
                .collect_results(&self.device, self.compute_data.current_compute_frame);
            self.gpu_timing.maybe_log();
        }

        self.compute_command_resource_manager.reset_command_pool(
            self.compute_command_resources.compute_pool_ids
                [self.compute_data.current_compute_frame],
        );
        self.compute_command_resource_manager.reset_command_pool(
            self.compute_command_resources.compute_transfer_pool_ids
                [self.compute_data.current_compute_frame],
        );
        self.compute_data.compute_staging_buffers[self.compute_data.current_compute_frame]
            .reset(&self.device);
    }

    pub fn dispatch_compute<G: ComputeSubmitGroup>(&mut self) {
        let group_type_id = TypeId::of::<G>();
        let fences = self
            .compute_data
            .compute_fences
            .entry(group_type_id)
            .or_insert_with(|| {
                std::array::from_fn(|_| {
                    let fence_create_info = vk::FenceCreateInfo {
                        s_type: vk::StructureType::FENCE_CREATE_INFO,
                        p_next: ptr::null(),
                        flags: vk::FenceCreateFlags::SIGNALED,
                        _marker: PhantomData,
                    };
                    unsafe {
                        self.device
                            .create_fence(&fence_create_info, None)
                            .expect("Failed to create compute fence")
                    }
                })
            });
        let fence = fences[self.compute_data.current_compute_frame];

        unsafe {
            let _ = self.device.reset_fences(&[fence]);
        }

        let command_buffer = self
            .compute_command_resource_manager
            .get_free_command_buffer(
                self.compute_command_resources.compute_pool_ids
                    [self.compute_data.current_compute_frame],
                vk::CommandBufferLevel::PRIMARY,
            );

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: ptr::null(),
            _marker: PhantomData,
        };
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin compute command buffer");
        }

        #[cfg(feature = "gpu-profiling")]
        if self
            .gpu_timing
            .query_count(self.compute_data.current_compute_frame)
            == 0
        {
            unsafe {
                self.device.cmd_reset_query_pool(
                    command_buffer,
                    self.gpu_timing
                        .query_pool(self.compute_data.current_compute_frame),
                    0,
                    crate::vulkan::gpu_timing::GpuTimingManager::max_timestamps_per_frame(),
                );
            }
        }

        let pipeline_manager = self.pipeline_manager.read().unwrap();
        let descriptor_manager = self.descriptor_manager.read().unwrap();
        let resource_manager = self.resource_manager.read().unwrap();
        let mut context = VkComputeRecordContext::new(
            &self.device,
            command_buffer,
            self.compute_data.current_compute_frame,
            &pipeline_manager,
            &descriptor_manager,
            &resource_manager,
            &mut self.compute_command_resource_manager,
            &self.compute_command_resources,
            &self.compute_data.compute_record_resource,
            #[cfg(feature = "gpu-profiling")]
            &mut self.gpu_timing,
        );
        G::Recordables::record(&mut context);

        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end compute command buffer");
        }

        let mutex_ids = <G::Mutex as GpuMutexList>::mutex_ids();
        let mut wait_semaphores = Vec::with_capacity(mutex_ids.len());
        let mut wait_stage_masks = Vec::with_capacity(mutex_ids.len());
        let mut wait_values = Vec::with_capacity(mutex_ids.len());
        let mut signal_semaphores = Vec::with_capacity(mutex_ids.len());
        let mut signal_values = Vec::with_capacity(mutex_ids.len());

        let mut gpu_mutexes = self.gpu_mutexes.lock().unwrap();
        for mutex_id in &mutex_ids {
            let (semaphore, wait_value, signal_value) =
                acquire_mutex_state(&mut gpu_mutexes, &self.device, *mutex_id);
            wait_semaphores.push(semaphore);
            wait_stage_masks.push(vk::PipelineStageFlags::COMPUTE_SHADER);
            wait_values.push(wait_value);
            signal_semaphores.push(semaphore);
            signal_values.push(signal_value);
        }
        drop(gpu_mutexes);

        let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
            s_type: vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_value_count: wait_values.len() as u32,
            p_wait_semaphore_values: wait_values.as_ptr(),
            signal_semaphore_value_count: signal_values.len() as u32,
            p_signal_semaphore_values: signal_values.as_ptr(),
            _marker: PhantomData,
        };
        let compute_submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: if mutex_ids.is_empty() {
                ptr::null()
            } else {
                (&timeline_submit_info as *const vk::TimelineSemaphoreSubmitInfo).cast()
            },
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stage_masks.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            _marker: PhantomData,
        };

        unsafe {
            let _ = self
                .device
                .queue_submit(self.compute_queue, &[compute_submit_info], fence);
        }
    }

    pub fn wait_for_compute<G: ComputeSubmitGroup>(&self) {
        if let Some(fences) = self.compute_data.compute_fences.get(&TypeId::of::<G>()) {
            unsafe {
                let _ = self.device.wait_for_fences(fences, true, u64::MAX);
            }
        }
    }

    pub fn upload_to_buffer<M: GpuMutex + 'static, T: BufferMarker<Lifetime = Persistent>, D>(
        &mut self,
        index: usize,
        upload_data: &[D],
        dst_byte_offset: u64,
    ) {
        let buffer_id = crate::vulkan::resource_config::buffer_id::<T>(index)
            .expect("Destination buffer not found");
        let is_host_visible = {
            let resource_manager = self.resource_manager.read().unwrap();
            resource_manager.is_buffer_host_visible(buffer_id)
        };

        if is_host_visible {
            let resource_manager = self.resource_manager.read().unwrap();
            resource_manager.copy_data_to_buffer(buffer_id, dst_byte_offset, upload_data);
        } else {
            self.upload_to_buffer_via_staging::<M, T, D>(index, upload_data, dst_byte_offset);
        }
    }

    fn upload_to_buffer_via_staging<
        M: GpuMutex + 'static,
        T: BufferMarker<Lifetime = Persistent>,
        D,
    >(
        &mut self,
        index: usize,
        upload_data: &[D],
        dst_byte_offset: u64,
    ) {
        let data_size = std::mem::size_of_val(upload_data) as u64;
        let buffer_id = crate::vulkan::resource_config::buffer_id::<T>(index)
            .expect("Destination buffer not found");
        let allocation = self.compute_data.compute_staging_buffers
            [self.compute_data.current_compute_frame]
            .allocate_and_write(
                upload_data,
                &self.device,
                self.physical_device,
                &self.instance,
                &self.queue_family_indices,
            );

        let (dst, dst_size) = {
            let resource_manager = self.resource_manager.read().unwrap();
            let dst = resource_manager.get_buffer(buffer_id);
            let dst_size = resource_manager.get_buffer_size(buffer_id);
            (
                dst.expect("Destination buffer not found"),
                dst_size.expect("Destination buffer size not found"),
            )
        };
        let size = data_size.min(dst_size.saturating_sub(dst_byte_offset));
        if size == 0 {
            return;
        }

        let cb = self
            .compute_command_resource_manager
            .get_free_command_buffer(
                self.compute_command_resources.compute_transfer_pool_ids
                    [self.compute_data.current_compute_frame],
                vk::CommandBufferLevel::PRIMARY,
            );
        begin_transfer_command_buffer(&self.device, cb);
        crate::vulkan::transfer_commands::copy_buffer(
            &self.device,
            cb,
            allocation.buffer,
            allocation.offset,
            dst,
            dst_byte_offset,
            size,
        );
        let timeline = if M::IS_NOOP {
            None
        } else {
            let mut gpu_mutexes = self.gpu_mutexes.lock().unwrap();
            Some(acquire_mutex_state(
                &mut gpu_mutexes,
                &self.device,
                TypeId::of::<M>(),
            ))
        };
        let completion_value = self.compute_transfer_next_timeline_value;
        self.compute_transfer_next_timeline_value += 1;
        self.compute_transfer_last_submitted_values[self.compute_data.current_compute_frame] =
            completion_value;
        submit_transfer_command_buffer(
            &self.device,
            self.compute_transfer_queue,
            cb,
            timeline.as_slice(),
            Some((self.compute_transfer_timeline, completion_value)),
        );
    }

    pub fn set_compute_record_resource<T: Clone + Send + Sync + 'static>(&mut self, value: T) {
        self.compute_data.compute_record_resource.insert(value);
    }

    pub fn read_buffer_data<T: BufferMarker<Lifetime = Persistent>, D: Copy, const N: usize>(
        &self,
        index: usize,
    ) -> Option<[D; N]> {
        let resource_manager = self.resource_manager.read().unwrap();
        resource_manager.read_buffer_data(
            crate::vulkan::resource_config::buffer_id::<T>(index)
                .filter(|id| *id != crate::vulkan::resource_config::BufferId::INVALID)?,
        )
    }
}

impl Drop for ComputeExecutor {
    fn drop(&mut self) {
        unsafe {
            self.device
                .queue_wait_idle(self.compute_queue)
                .expect("Failed to wait for compute queue idle");
            if self.compute_transfer_queue != self.compute_queue {
                self.device
                    .queue_wait_idle(self.compute_transfer_queue)
                    .expect("Failed to wait for compute transfer queue idle");
            }
            for fences in self.compute_data.compute_fences.values() {
                for fence in fences {
                    self.device.destroy_fence(*fence, None);
                }
            }
            self.device
                .destroy_semaphore(self.compute_transfer_timeline, None);
        }

        #[cfg(feature = "gpu-profiling")]
        self.gpu_timing.destroy(&self.device);

        self.compute_data.compute_fences.clear();

        for staging_buffer in &mut self.compute_data.compute_staging_buffers {
            staging_buffer.destroy(&self.device);
        }

        self.compute_command_resource_manager.cleanup();
    }
}

unsafe impl Send for ComputeExecutor {}
