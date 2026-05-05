#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SVOCounters {
    pub node_count: u32,
    pub leaf_count: u32,
    pub free_node_count: u32,
    pub free_leaf_count: u32,
}
