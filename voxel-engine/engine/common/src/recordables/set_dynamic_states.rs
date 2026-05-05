use crate::vulkan::record_context::VkGraphicsRecordContext;
use crate::vulkan::recordable::VkGraphicsRecordable;

#[derive(Clone)]
pub struct SetDynamicStates {}

impl VkGraphicsRecordable for SetDynamicStates {
    fn record(context: &mut VkGraphicsRecordContext) {
        let (width, height) = context.get_swapchain_extent();
        context.cmd_set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        context.cmd_set_scissor(0, 0, width, height);
    }
}
