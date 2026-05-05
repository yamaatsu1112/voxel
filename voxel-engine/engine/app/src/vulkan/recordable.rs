use crate::vulkan::record_context::{VkComputeRecordContext, VkGraphicsRecordContext};

pub trait VkGraphicsRecordable {
    fn record(context: &mut VkGraphicsRecordContext);
}

macro_rules! impl_graphics_recordable_for_tuple {
    ($($T:ident),+) => {
        impl<$($T: VkGraphicsRecordable),+> VkGraphicsRecordable for ($($T,)+) {
            fn record(context: &mut VkGraphicsRecordContext) {
                $($T::record(context);)+
            }
        }
    };
}

impl_graphics_recordable_for_tuple!(A);
impl_graphics_recordable_for_tuple!(A, B);
impl_graphics_recordable_for_tuple!(A, B, C);
impl_graphics_recordable_for_tuple!(A, B, C, D);
impl_graphics_recordable_for_tuple!(A, B, C, D, E);

pub trait VkComputeRecordable: Send {
    fn record(context: &mut VkComputeRecordContext);
}

macro_rules! impl_compute_recordable_for_tuple {
    ($($T:ident),+) => {
        impl<$($T: VkComputeRecordable),+> VkComputeRecordable for ($($T,)+) {
            fn record(context: &mut VkComputeRecordContext) {
                $($T::record(context);)+
            }
        }
    };
}

impl_compute_recordable_for_tuple!(A);
impl_compute_recordable_for_tuple!(A, B);
impl_compute_recordable_for_tuple!(A, B, C);
impl_compute_recordable_for_tuple!(A, B, C, D);
impl_compute_recordable_for_tuple!(A, B, C, D, E);
