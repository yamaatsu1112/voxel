use std::sync::atomic::Ordering;

use crate::vulkan::resource_config::ReusableCommandBufferMarker;

mod vulkan {
    pub mod resource_config {
        use std::sync::atomic::AtomicUsize;

        pub trait ReusableCommandBufferMarker: 'static {
            const COUNT: usize;
            fn ids_slice() -> &'static [AtomicUsize];
        }
    }

    pub mod vulkan_renderer_core {
        pub const MAX_COMPUTE_FRAMES_IN_FLIGHT: usize = 2;
    }
}

const TEST_COUNT: usize = 3;

engine_macro::define_reusable_command_buffer! {
    pub struct TestReusableCommandBuffer;
    count = TEST_COUNT;
}

#[test]
fn reusable_command_buffer_marker_uses_frame_scaled_storage() {
    let ids = TestReusableCommandBuffer::ids_slice();

    assert_eq!(TestReusableCommandBuffer::COUNT, TEST_COUNT);
    assert_eq!(
        ids.len(),
        TEST_COUNT * vulkan::vulkan_renderer_core::MAX_COMPUTE_FRAMES_IN_FLIGHT
    );
}

#[test]
fn reusable_command_buffer_marker_initializes_all_slots_as_unallocated() {
    let ids = TestReusableCommandBuffer::ids_slice();

    assert!(
        ids.iter()
            .all(|id| id.load(Ordering::Acquire) == usize::MAX)
    );

    ids[1].store(7, Ordering::Release);
    assert_eq!(ids[1].load(Ordering::Acquire), 7);
    ids[1].store(usize::MAX, Ordering::Release);
}
