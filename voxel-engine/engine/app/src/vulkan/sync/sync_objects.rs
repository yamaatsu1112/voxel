use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use crate::vulkan::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT;

pub struct SyncObjects {
    pub image_available_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: [vk::Fence; MAX_FRAMES_IN_FLIGHT],
    pub graphics_transfer_timeline: vk::Semaphore,
}

impl SyncObjects {
    pub fn new(device: &ash::Device, swapchain_image_count: usize) -> Self {
        let fence_create_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
            _marker: PhantomData,
        };

        let in_flight_fences: [vk::Fence; MAX_FRAMES_IN_FLIGHT] = std::array::from_fn(|_| unsafe {
            device
                .create_fence(&fence_create_info, None)
                .expect("Failed to create fence")
        });

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
            _marker: PhantomData,
        };

        let render_finished_semaphores: Vec<vk::Semaphore> = (0..swapchain_image_count)
            .map(|_| unsafe {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create semaphore")
            })
            .collect();

        let image_available_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT] =
            std::array::from_fn(|_| unsafe {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create semaphore")
            });

        let graphics_transfer_timeline = create_timeline_semaphore(device);

        Self {
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            graphics_transfer_timeline,
        }
    }
}

pub fn create_timeline_semaphore(device: &ash::Device) -> vk::Semaphore {
    let mut semaphore_type_create_info = vk::SemaphoreTypeCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO,
        p_next: ptr::null(),
        semaphore_type: vk::SemaphoreType::TIMELINE,
        initial_value: 0,
        _marker: PhantomData,
    };
    let semaphore_create_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: (&mut semaphore_type_create_info as *mut vk::SemaphoreTypeCreateInfo).cast(),
        flags: vk::SemaphoreCreateFlags::empty(),
        _marker: PhantomData,
    };

    unsafe {
        device
            .create_semaphore(&semaphore_create_info, None)
            .expect("Failed to create timeline semaphore")
    }
}
