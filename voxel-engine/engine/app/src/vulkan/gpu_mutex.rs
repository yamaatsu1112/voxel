use std::marker::PhantomData;
use std::ptr;

use ash::vk;

pub(crate) struct GpuMutexState {
    pub semaphore: vk::Semaphore,
    pub value: u64,
}

impl GpuMutexState {
    pub fn new(device: &ash::Device) -> Self {
        let timeline_info = vk::SemaphoreTypeCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO,
            p_next: ptr::null(),
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: 0,
            _marker: PhantomData,
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: (&timeline_info as *const vk::SemaphoreTypeCreateInfo).cast(),
            flags: vk::SemaphoreCreateFlags::empty(),
            _marker: PhantomData,
        };

        let semaphore = unsafe {
            device
                .create_semaphore(&semaphore_create_info, None)
                .expect("Failed to create timeline semaphore")
        };

        Self {
            semaphore,
            value: 0,
        }
    }

    pub fn acquire(&mut self) -> (u64, u64) {
        let wait_value = self.value;
        let signal_value = self
            .value
            .checked_add(1)
            .expect("GpuMutex timeline value overflowed");
        self.value = signal_value;
        (wait_value, signal_value)
    }
}
