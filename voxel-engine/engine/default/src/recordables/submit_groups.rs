use crate::rendering::NoMutex;
use crate::vulkan::GraphicsSubmitGroup;

use super::{
    BeginColorGBufferRendering, BeginDynamicColorGBufferRendering, BeginDynamicGBufferRendering,
    BeginGBufferRendering, BeginSwapchainRendering, DynamicGBufferColorWritePipelineRecordable,
    DynamicGBufferWritePipelineRecordable, EndRendering, EndRenderingPhase,
    GBufferColorWritePipelineRecordable, GBufferWritePipelineRecordable, MainPipelineRecordable,
    SetDynamicStates, TransitDynamicGBufferColorImage, TransitDynamicGBufferImage,
    TransitGBufferColorImage, TransitGBufferImage, UIPipelineRecordable,
};

#[derive(Clone)]
pub struct RenderSubmitGroup;

impl GraphicsSubmitGroup for RenderSubmitGroup {
    type Mutex = NoMutex;
    type Recordables = (
        (
            BeginGBufferRendering,
            SetDynamicStates,
            GBufferWritePipelineRecordable,
            EndRenderingPhase,
            TransitGBufferImage,
        ),
        (
            BeginColorGBufferRendering,
            SetDynamicStates,
            GBufferColorWritePipelineRecordable,
            EndRenderingPhase,
            TransitGBufferColorImage,
        ),
        (
            BeginDynamicGBufferRendering,
            SetDynamicStates,
            DynamicGBufferWritePipelineRecordable,
            EndRenderingPhase,
            TransitDynamicGBufferImage,
        ),
        (
            BeginDynamicColorGBufferRendering,
            SetDynamicStates,
            DynamicGBufferColorWritePipelineRecordable,
            EndRenderingPhase,
            TransitDynamicGBufferColorImage,
        ),
        (
            BeginSwapchainRendering,
            SetDynamicStates,
            MainPipelineRecordable,
            UIPipelineRecordable,
            EndRendering,
        ),
    );
}
