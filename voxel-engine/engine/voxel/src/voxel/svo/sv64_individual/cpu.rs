use super::super::backend::{SVOBackend, cast_slice_to_bytes};
use super::super::counters::SVOCounters;
use super::super::{LEAF_VOXEL_COUNT_EXP, WORLD_VOXEL_COUNT, WORLD_VOXEL_COUNT_EXP};
use core::mem::size_of;

const ROOT_NODE_INDEX: usize = 0;
const SV64_INDIVIDUAL_MAX_NODE_COUNT: u32 = 65536;
const SV64_INDIVIDUAL_MAX_LEAF_COUNT: u32 = 262144;
const SV64_INDIVIDUAL_LEAF_VOXEL_COUNT: u32 = 1 << LEAF_VOXEL_COUNT_EXP;
const SV64_BRANCH_FACTOR_EXP: u32 = 2;
const SV64_INDIVIDUAL_MAX_DEPTH: u32 =
    (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / SV64_BRANCH_FACTOR_EXP;
const SV64_CHILDREN_PER_NODE: usize = 1 << (SV64_BRANCH_FACTOR_EXP * 3);

const CHILD_INDEX_OFFSET: u32 = 0;
const CHILD_MASK_OFFSET: u32 = 30;
const FILLED_OFFSET: u32 = 31;
const CHILD_INDEX_MASK: u32 = 0x3FFF_FFFFu32;
const CHILD_MASK_BIT: u32 = 1u32 << CHILD_MASK_OFFSET;
const FILLED_BIT: u32 = 1u32 << FILLED_OFFSET;

#[repr(transparent)]
#[derive(Copy, Clone)]
struct ChildInfo(u32);

impl ChildInfo {
    fn new() -> Self {
        Self(0)
    }

    fn has_child(self) -> bool {
        (self.0 & CHILD_MASK_BIT) != 0
    }

    fn is_filled(self) -> bool {
        (self.0 & FILLED_BIT) != 0
    }

    fn get_child_index(self) -> u32 {
        (self.0 & CHILD_INDEX_MASK) >> CHILD_INDEX_OFFSET
    }

    fn set_child_mask(&mut self, value: bool) {
        if value {
            self.0 |= CHILD_MASK_BIT;
        } else {
            self.0 &= !CHILD_MASK_BIT;
        }
    }

    fn set_filled(&mut self, value: bool) {
        if value {
            self.0 |= FILLED_BIT;
        } else {
            self.0 &= !FILLED_BIT;
        }
    }

    fn set_child_index(&mut self, index: u32) {
        self.0 = (self.0 & !CHILD_INDEX_MASK) | ((index << CHILD_INDEX_OFFSET) & CHILD_INDEX_MASK);
    }
}

#[repr(C)]
#[derive(Clone)]
struct SV64Node {
    child_data: [ChildInfo; SV64_CHILDREN_PER_NODE],
}

impl SV64Node {
    fn new() -> Self {
        Self {
            child_data: [ChildInfo::new(); SV64_CHILDREN_PER_NODE],
        }
    }

    fn has_child(&self, child_index: usize) -> bool {
        self.child_data[child_index].has_child()
    }

    fn is_filled(&self, child_index: usize) -> bool {
        self.child_data[child_index].is_filled()
    }

    fn get_child_index(&self, child_index: usize) -> u32 {
        self.child_data[child_index].get_child_index()
    }

    fn set_child_mask(&mut self, child_index: usize, value: bool) {
        self.child_data[child_index].set_child_mask(value);
    }

    fn set_filled(&mut self, child_index: usize, value: bool) {
        self.child_data[child_index].set_filled(value);
    }

    fn set_child_index(&mut self, child_index: usize, index: u32) {
        self.child_data[child_index].set_child_index(index);
    }
}

#[repr(C)]
#[derive(Clone)]
struct SVOLeaf {
    voxel_data: u64,
}

impl SVOLeaf {
    fn new() -> Self {
        Self { voxel_data: 0 }
    }
}

#[derive(Clone)]
pub struct SV64Individual {
    nodes: Vec<SV64Node>,
    leaves: Vec<SVOLeaf>,
    counters: SVOCounters,
    free_node_indices: Vec<u32>,
    free_leaf_indices: Vec<u32>,
}

pub(crate) const GPU_NODE_BYTE_SIZE: u64 = size_of::<SV64Node>() as u64;
pub(crate) const GPU_LEAF_BYTE_SIZE: u64 = size_of::<SVOLeaf>() as u64;

impl Default for SV64Individual {
    fn default() -> Self {
        <Self as SVOBackend>::new()
    }
}

impl SVOBackend for SV64Individual {
    const GPU_NODE_BYTE_SIZE: u64 = GPU_NODE_BYTE_SIZE;
    const GPU_LEAF_BYTE_SIZE: u64 = GPU_LEAF_BYTE_SIZE;
    const MAX_NODE_COUNT: u32 = SV64_INDIVIDUAL_MAX_NODE_COUNT;
    const MAX_LEAF_COUNT: u32 = SV64_INDIVIDUAL_MAX_LEAF_COUNT;
    const MAX_DEPTH: u32 = SV64_INDIVIDUAL_MAX_DEPTH;
    const LEAF_VOXEL_COUNT: u32 = SV64_INDIVIDUAL_LEAF_VOXEL_COUNT;
    const BRANCH_FACTOR_EXP: u32 = SV64_BRANCH_FACTOR_EXP;

    fn new() -> Self {
        Self {
            nodes: vec![SV64Node::new(); SV64_INDIVIDUAL_MAX_NODE_COUNT as usize],
            leaves: vec![SVOLeaf::new(); SV64_INDIVIDUAL_MAX_LEAF_COUNT as usize],
            counters: SVOCounters {
                node_count: 1,
                leaf_count: 0,
                free_node_count: 0,
                free_leaf_count: 0,
            },
            free_node_indices: vec![0; SV64_INDIVIDUAL_MAX_NODE_COUNT as usize],
            free_leaf_indices: vec![0; SV64_INDIVIDUAL_MAX_LEAF_COUNT as usize],
        }
    }

    fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: bool) {
        self.set_voxel_impl(x, y, z, value);
    }

    fn get_voxel(&self, x: i32, y: i32, z: i32) -> bool {
        self.get_voxel_impl(x, y, z)
    }

    fn nodes_as_bytes(&self) -> &[u8] {
        cast_slice_to_bytes(&self.nodes)
    }

    fn leaves_as_bytes(&self) -> &[u8] {
        cast_slice_to_bytes(&self.leaves)
    }

    fn counters(&self) -> &SVOCounters {
        &self.counters
    }

    fn free_node_indices_as_bytes(&self) -> &[u8] {
        cast_slice_to_bytes(&self.free_node_indices)
    }

    fn free_leaf_indices_as_bytes(&self) -> &[u8] {
        cast_slice_to_bytes(&self.free_leaf_indices)
    }
}

impl SV64Individual {
    fn allocate_node(&mut self) -> usize {
        let index = if self.counters.free_node_count > 0 {
            let index = self.free_node_indices[(self.counters.free_node_count - 1) as usize];
            self.counters.free_node_count -= 1;
            index as usize
        } else {
            let index = self.counters.node_count as usize;
            self.counters.node_count += 1;
            index
        };

        if index >= Self::MAX_NODE_COUNT as usize {
            panic!("SVO: Out of nodes");
        }

        self.nodes[index] = SV64Node::new();
        index
    }

    fn allocate_leaf(&mut self) -> usize {
        let index = if self.counters.free_leaf_count > 0 {
            let index = self.free_leaf_indices[(self.counters.free_leaf_count - 1) as usize];
            self.counters.free_leaf_count -= 1;
            index as usize
        } else {
            let index = self.counters.leaf_count as usize;
            self.counters.leaf_count += 1;
            index
        };

        if index >= Self::MAX_LEAF_COUNT as usize {
            panic!("SVO: Out of leaves");
        }

        self.leaves[index] = SVOLeaf::new();
        index
    }

    fn free_node(&mut self, index: usize) {
        if index >= Self::MAX_NODE_COUNT as usize {
            panic!("SVO: Index out of bounds");
        }
        self.free_node_indices[self.counters.free_node_count as usize] = index as u32;
        self.counters.free_node_count += 1;
    }

    fn free_leaf(&mut self, index: usize) {
        if index >= Self::MAX_LEAF_COUNT as usize {
            panic!("SVO: Index out of bounds");
        }
        self.free_leaf_indices[self.counters.free_leaf_count as usize] = index as u32;
        self.counters.free_leaf_count += 1;
    }

    fn get_child_index(&self, x: i32, y: i32, z: i32, depth: u32) -> usize {
        let shifted_x = x + (WORLD_VOXEL_COUNT as i32 >> 1);
        let shifted_y = y + (WORLD_VOXEL_COUNT as i32 >> 1);
        let shifted_z = z + (WORLD_VOXEL_COUNT as i32 >> 1);

        let shift = (Self::MAX_DEPTH - depth - 1) * Self::BRANCH_FACTOR_EXP + LEAF_VOXEL_COUNT_EXP;
        let mask = (1 << Self::BRANCH_FACTOR_EXP) - 1;

        let x_bits = (shifted_x >> shift) & mask;
        let y_bits = (shifted_y >> shift) & mask;
        let z_bits = (shifted_z >> shift) & mask;

        (x_bits | (y_bits << Self::BRANCH_FACTOR_EXP) | (z_bits << (Self::BRANCH_FACTOR_EXP * 2)))
            as usize
    }

    fn get_leaf_offset(&self, x: i32, y: i32, z: i32) -> usize {
        let shifted_x = x + (WORLD_VOXEL_COUNT as i32 >> 1);
        let shifted_y = y + (WORLD_VOXEL_COUNT as i32 >> 1);
        let shifted_z = z + (WORLD_VOXEL_COUNT as i32 >> 1);

        let x_offset = shifted_x % Self::LEAF_VOXEL_COUNT as i32;
        let y_offset = shifted_y % Self::LEAF_VOXEL_COUNT as i32;
        let z_offset = shifted_z % Self::LEAF_VOXEL_COUNT as i32;

        (x_offset
            + y_offset * Self::LEAF_VOXEL_COUNT as i32
            + z_offset * Self::LEAF_VOXEL_COUNT as i32 * Self::LEAF_VOXEL_COUNT as i32)
            as usize
    }

    fn set_voxel_impl(&mut self, x: i32, y: i32, z: i32, value: bool) {
        let self_ptr = self as *mut Self;
        let mut node_index = ROOT_NODE_INDEX;
        let mut child_index;
        let mut node_indices = [0; Self::MAX_DEPTH as usize];
        let mut child_indices = [0; Self::MAX_DEPTH as usize];

        for depth in 0..Self::MAX_DEPTH - 1 {
            let node = unsafe { &mut (&mut (*self_ptr).nodes)[node_index] };
            child_index = self.get_child_index(x, y, z, depth);
            node_indices[depth as usize] = node_index;
            child_indices[depth as usize] = child_index;

            if !node.has_child(child_index) {
                if node.is_filled(child_index) == value {
                    return;
                }

                let new_node_index = self.allocate_node();
                if new_node_index == 0 {
                    panic!("SVO: Out of space");
                }

                node.set_child_mask(child_index, true);
                node.set_child_index(child_index, new_node_index as u32);
                node_index = new_node_index;

                let new_node = unsafe { &mut (&mut (*self_ptr).nodes)[node_index] };
                for i in 0..SV64_CHILDREN_PER_NODE {
                    new_node.set_filled(i, node.is_filled(child_index));
                }
            } else {
                node_index = node.get_child_index(child_index) as usize;
            }
        }

        let node = unsafe { &mut (&mut (*self_ptr).nodes)[node_index] };
        child_index = self.get_child_index(x, y, z, Self::MAX_DEPTH - 1);
        node_indices[Self::MAX_DEPTH as usize - 1] = node_index;
        child_indices[Self::MAX_DEPTH as usize - 1] = child_index;

        if !node.has_child(child_index) {
            if node.is_filled(child_index) == value {
                return;
            }

            let new_leaf_index = self.allocate_leaf();
            if new_leaf_index == Self::MAX_LEAF_COUNT as usize - 1 {
                panic!("SVO: Out of space");
            }

            node.set_child_mask(child_index, true);
            node.set_child_index(child_index, new_leaf_index as u32);
            node_index = new_leaf_index;

            let leaf = unsafe { &mut (&mut (*self_ptr).leaves)[node_index] };
            leaf.voxel_data = if node.is_filled(child_index) {
                u64::MAX
            } else {
                0
            };
        } else {
            node_index = node.get_child_index(child_index) as usize;
        }

        let leaf = unsafe { &mut (&mut (*self_ptr).leaves)[node_index] };
        let bit_offset = self.get_leaf_offset(x, y, z);
        let mask = 1u64 << bit_offset;

        if value {
            leaf.voxel_data |= mask;
        } else {
            leaf.voxel_data &= !mask;
        }

        self.free_nodes_to_be_freed(&node_indices, &child_indices, node_index);
    }

    fn free_nodes_to_be_freed(
        &mut self,
        node_indices: &[usize; Self::MAX_DEPTH as usize],
        child_indices: &[usize; Self::MAX_DEPTH as usize],
        leaf_index: usize,
    ) {
        let leaf_data = self.leaves[leaf_index].voxel_data;
        if leaf_data != 0 && leaf_data != u64::MAX {
            return;
        }

        let filled = leaf_data == u64::MAX;
        self.free_leaf(leaf_index);

        for i in (0..Self::MAX_DEPTH as usize).rev() {
            let node_index = node_indices[i];
            let child_index = child_indices[i];
            let node = &mut self.nodes[node_index];

            node.set_child_mask(child_index, false);
            node.set_child_index(child_index, 0);
            node.set_filled(child_index, filled);

            let mut has_any_children = false;
            let mut all_filled = true;
            let mut all_empty = true;
            for j in 0..SV64_CHILDREN_PER_NODE {
                if node.has_child(j) {
                    has_any_children = true;
                    break;
                }
                if node.is_filled(j) {
                    all_empty = false;
                }
                if !node.is_filled(j) {
                    all_filled = false;
                }
            }

            if has_any_children {
                break;
            }

            if filled {
                if !all_filled {
                    break;
                }
            } else if !all_empty {
                break;
            }

            if node_index == ROOT_NODE_INDEX {
                break;
            }

            self.free_node(node_index);
        }
    }

    fn get_voxel_impl(&self, x: i32, y: i32, z: i32) -> bool {
        let self_ptr = self as *const Self;
        let mut node_index = ROOT_NODE_INDEX;
        let mut child_index;

        for depth in 0..Self::MAX_DEPTH - 1 {
            let node = unsafe { &(&(*self_ptr).nodes)[node_index] };
            child_index = self.get_child_index(x, y, z, depth);

            if !node.has_child(child_index) {
                return node.is_filled(child_index);
            }
            node_index = node.get_child_index(child_index) as usize;
        }

        let node = unsafe { &(&(*self_ptr).nodes)[node_index] };
        child_index = self.get_child_index(x, y, z, Self::MAX_DEPTH - 1);
        if !node.has_child(child_index) {
            return node.is_filled(child_index);
        }
        node_index = node.get_child_index(child_index) as usize;

        let bit_offset = self.get_leaf_offset(x, y, z);
        let leaf = unsafe { &(&(*self_ptr).leaves)[node_index] };
        let mask = 1u64 << bit_offset;

        (leaf.voxel_data & mask) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::svo::get_leaf_coords;

    fn node_count_in_use(svo: &SV64Individual) -> u32 {
        svo.counters().node_count - svo.counters().free_node_count
    }

    fn leaf_count_in_use(svo: &SV64Individual) -> u32 {
        svo.counters().leaf_count - svo.counters().free_leaf_count
    }

    #[test]
    fn test_child_index_extracts_two_bits_per_axis() {
        let svo = SV64Individual::new();

        assert_eq!(svo.get_child_index(0, 0, 0, 0), 42);
        assert_eq!(svo.get_child_index(-512, -512, -512, 0), 0);
        assert_eq!(svo.get_child_index(508, 508, 508, 3), 63);
    }

    #[test]
    fn test_leaf_coords_match_world_size() {
        assert_eq!(get_leaf_coords(0, 0, 0), (128, 128, 128));
        assert_eq!(get_leaf_coords(-512, -512, -512), (0, 0, 0));
        assert_eq!(get_leaf_coords(511, 511, 511), (255, 255, 255));
    }

    #[test]
    fn test_set_and_get_voxel_across_edges() {
        let mut svo = SV64Individual::new();
        let coords = [
            (0, 0, 0),
            (-512, -512, -512),
            (511, 511, 511),
            (-1, 17, -300),
        ];

        for &(x, y, z) in &coords {
            svo.set_voxel(x, y, z, true);
            assert!(svo.get_voxel(x, y, z));
            svo.set_voxel(x, y, z, false);
            assert!(!svo.get_voxel(x, y, z));
        }
    }

    #[test]
    fn test_gpu_accessors_match_sv64_layout_lengths() {
        let svo = SV64Individual::new();

        assert_eq!(
            svo.nodes_as_bytes().len(),
            SV64_INDIVIDUAL_MAX_NODE_COUNT as usize * size_of::<SV64Node>()
        );
        assert_eq!(
            svo.leaves_as_bytes().len(),
            SV64_INDIVIDUAL_MAX_LEAF_COUNT as usize * size_of::<SVOLeaf>()
        );
        assert_eq!(
            svo.free_node_indices_as_bytes().len(),
            SV64_INDIVIDUAL_MAX_NODE_COUNT as usize * size_of::<u32>()
        );
        assert_eq!(
            svo.free_leaf_indices_as_bytes().len(),
            SV64_INDIVIDUAL_MAX_LEAF_COUNT as usize * size_of::<u32>()
        );
    }

    #[test]
    fn test_collapse_fully_filled_leaf_back_into_parent_bit() {
        let mut svo = SV64Individual::new();

        for z in 0..SV64Individual::LEAF_VOXEL_COUNT as i32 {
            for y in 0..SV64Individual::LEAF_VOXEL_COUNT as i32 {
                for x in 0..SV64Individual::LEAF_VOXEL_COUNT as i32 {
                    svo.set_voxel(x, y, z, true);
                }
            }
        }

        assert!(svo.get_voxel(0, 0, 0));
        assert!(svo.get_voxel(3, 3, 3));
        assert_eq!(leaf_count_in_use(&svo), 0);
        assert_eq!(node_count_in_use(&svo), SV64Individual::MAX_DEPTH);
    }

    #[test]
    fn test_collapse_fully_emptied_leaf_back_into_parent_bit() {
        let mut svo = SV64Individual::new();

        svo.set_voxel(0, 0, 0, true);
        svo.set_voxel(1, 0, 0, true);
        svo.set_voxel(0, 1, 0, true);
        assert_eq!(leaf_count_in_use(&svo), 1);

        svo.set_voxel(0, 0, 0, false);
        svo.set_voxel(1, 0, 0, false);
        svo.set_voxel(0, 1, 0, false);

        assert!(!svo.get_voxel(0, 0, 0));
        assert!(!svo.get_voxel(1, 0, 0));
        assert!(!svo.get_voxel(0, 1, 0));
        assert_eq!(leaf_count_in_use(&svo), 0);
        assert_eq!(node_count_in_use(&svo), 1);
    }
}
