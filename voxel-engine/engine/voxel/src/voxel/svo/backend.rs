use super::counters::SVOCounters;
use core::mem::size_of_val;
use core::slice;

pub(crate) fn cast_slice_to_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) }
}

pub trait SVOBackend: Clone + Default + Send + Sync {
    const GPU_NODE_BYTE_SIZE: u64;
    const GPU_LEAF_BYTE_SIZE: u64;
    const MAX_NODE_COUNT: u32;
    const MAX_LEAF_COUNT: u32;
    const MAX_DEPTH: u32;
    const LEAF_VOXEL_COUNT: u32;
    const BRANCH_FACTOR_EXP: u32;

    fn new() -> Self;
    fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: bool);
    fn get_voxel(&self, x: i32, y: i32, z: i32) -> bool;
    fn nodes_as_bytes(&self) -> &[u8];
    fn leaves_as_bytes(&self) -> &[u8];
    fn counters(&self) -> &SVOCounters;
    fn free_node_indices_as_bytes(&self) -> &[u8];
    fn free_leaf_indices_as_bytes(&self) -> &[u8];
}
