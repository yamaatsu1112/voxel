use crate::vulkan::{ErasedGraphicsSubmitGroup, GraphicsSubmitGroup};

#[derive(Default)]
pub struct RenderGraph {
    submit_groups: Vec<Box<dyn ErasedGraphicsSubmitGroup>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_pass<G: GraphicsSubmitGroup>(&mut self, group: G) {
        self.submit_groups.push(Box::new(group));
    }

    pub(crate) fn take(&mut self) -> Vec<Box<dyn ErasedGraphicsSubmitGroup>> {
        std::mem::take(&mut self.submit_groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rendering::GpuMutex;
    use crate::vulkan::record_context::VkGraphicsRecordContext;
    use crate::vulkan::recordable::VkGraphicsRecordable;
    use std::any::TypeId;

    struct FirstMutex;
    impl GpuMutex for FirstMutex {}

    struct SecondMutex;
    impl GpuMutex for SecondMutex {}

    struct NoopRecordable;

    impl VkGraphicsRecordable for NoopRecordable {
        fn record(_context: &mut VkGraphicsRecordContext) {}
    }

    #[derive(Clone)]
    struct FirstPass;

    impl GraphicsSubmitGroup for FirstPass {
        type Mutex = FirstMutex;
        type Recordables = (NoopRecordable,);
    }

    #[derive(Clone)]
    struct SecondPass;

    impl GraphicsSubmitGroup for SecondPass {
        type Mutex = SecondMutex;
        type Recordables = (NoopRecordable,);
    }

    #[test]
    fn take_returns_groups_in_registration_order() {
        let mut graph = RenderGraph::new();
        graph.add_pass(FirstPass);
        graph.add_pass(SecondPass);

        let groups = graph.take();

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].mutex_ids(), vec![TypeId::of::<FirstMutex>()]);
        assert_eq!(groups[1].mutex_ids(), vec![TypeId::of::<SecondMutex>()]);
    }

    #[test]
    fn take_clears_graph() {
        let mut graph = RenderGraph::new();
        graph.add_pass(FirstPass);

        let groups = graph.take();

        assert_eq!(groups.len(), 1);
        assert!(graph.take().is_empty());
    }

    #[test]
    fn take_returns_empty_when_no_passes_are_registered() {
        let mut graph = RenderGraph::new();

        assert!(graph.take().is_empty());
    }
}
