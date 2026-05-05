use super::super::backend::{SVOBackend, cast_slice_to_bytes};
use super::super::counters::SVOCounters;
use super::super::{LEAF_VOXEL_COUNT_EXP, WORLD_VOXEL_COUNT, WORLD_VOXEL_COUNT_EXP};
use core::mem::size_of;

const ROOT_NODE_INDEX: usize = 0;
const SVO_INDIVIDUAL_MAX_NODE_COUNT: u32 = 131072;
const SVO_INDIVIDUAL_MAX_LEAF_COUNT: u32 = 524288;
const SVO_INDIVIDUAL_BRANCH_FACTOR_EXP: u32 = 1;
const SVO_INDIVIDUAL_MAX_DEPTH: u32 =
    (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / SVO_INDIVIDUAL_BRANCH_FACTOR_EXP;
const SVO_INDIVIDUAL_LEAF_VOXEL_COUNT: u32 = 1 << LEAF_VOXEL_COUNT_EXP;

// Constants for ChildInfo bit manipulation
// Bit layout: [1bit filled][1bit child_mask][30bit child_index]
const CHILD_INDEX_OFFSET: u32 = 0;
const CHILD_MASK_OFFSET: u32 = 30;
const FILLED_OFFSET: u32 = 31;
const CHILD_INDEX_MASK: u32 = 0x3FFFFFFFu32; // bits 0-29
const CHILD_MASK_BIT: u32 = 1u32 << CHILD_MASK_OFFSET; // 0x40000000
const FILLED_BIT: u32 = 1u32 << FILLED_OFFSET; // 0x80000000

// ChildInfo structure - stores child_mask, filled, and child_index in a single u32
// Bit layout: [1bit filled][1bit child_mask][30bit child_index]
#[repr(transparent)]
#[derive(Copy, Clone)]
struct ChildInfo(u32);

impl ChildInfo {
    pub(crate) fn new() -> Self {
        Self(0)
    }

    pub(crate) fn has_child(&self) -> bool {
        (self.0 & CHILD_MASK_BIT) != 0
    }

    pub(crate) fn is_filled(&self) -> bool {
        (self.0 & FILLED_BIT) != 0
    }

    pub(crate) fn get_child_index(&self) -> u32 {
        (self.0 & CHILD_INDEX_MASK) >> CHILD_INDEX_OFFSET
    }

    pub(crate) fn set_child_mask(&mut self, value: bool) {
        if value {
            self.0 |= CHILD_MASK_BIT;
        } else {
            self.0 &= !CHILD_MASK_BIT;
        }
    }

    pub(crate) fn set_filled(&mut self, value: bool) {
        if value {
            self.0 |= FILLED_BIT;
        } else {
            self.0 &= !FILLED_BIT;
        }
    }

    pub(crate) fn set_child_index(&mut self, index: u32) {
        // Clear the child_index bits and set new value
        self.0 = (self.0 & !CHILD_INDEX_MASK) | ((index << CHILD_INDEX_OFFSET) & CHILD_INDEX_MASK);
    }
}

// SVO Node structure - must match SVO.slang
#[repr(C)]
#[derive(Clone)]
pub(crate) struct SVONode {
    child_data: [ChildInfo; 8], // Each ChildInfo stores child_mask, filled, and child_index
}

impl SVONode {
    pub(crate) fn new() -> Self {
        Self {
            child_data: [ChildInfo::new(); 8],
        }
    }

    pub(crate) fn has_child(&self, child_index: usize) -> bool {
        self.child_data[child_index].has_child()
    }

    pub(crate) fn is_filled(&self, child_index: usize) -> bool {
        self.child_data[child_index].is_filled()
    }

    pub(crate) fn get_child_index(&self, child_index: usize) -> u32 {
        self.child_data[child_index].get_child_index()
    }

    pub(crate) fn set_child_mask(&mut self, child_index: usize, value: bool) {
        self.child_data[child_index].set_child_mask(value);
    }

    pub(crate) fn set_filled(&mut self, child_index: usize, value: bool) {
        self.child_data[child_index].set_filled(value);
    }

    pub(crate) fn set_child_index(&mut self, child_index: usize, index: u32) {
        self.child_data[child_index].set_child_index(index);
    }
}

// SVO Leaf structure - must match SVO.slang
#[repr(C)]
#[derive(Clone)]
pub(crate) struct SVOLeaf {
    voxel_data: u64,
}

impl SVOLeaf {
    pub(crate) fn new() -> Self {
        Self { voxel_data: 0 }
    }
}

// SVO data buffer layout - must match SVO.slang
#[derive(Clone)]
pub struct SVOIndividual {
    nodes: Vec<SVONode>,
    leaves: Vec<SVOLeaf>,
    counters: SVOCounters,
    free_node_indices: Vec<u32>,
    free_leaf_indices: Vec<u32>,
}

pub(crate) const GPU_NODE_BYTE_SIZE: u64 = size_of::<SVONode>() as u64;
pub(crate) const GPU_LEAF_BYTE_SIZE: u64 = size_of::<SVOLeaf>() as u64;

impl Default for SVOIndividual {
    fn default() -> Self {
        <Self as SVOBackend>::new()
    }
}

impl SVOBackend for SVOIndividual {
    const GPU_NODE_BYTE_SIZE: u64 = GPU_NODE_BYTE_SIZE;
    const GPU_LEAF_BYTE_SIZE: u64 = GPU_LEAF_BYTE_SIZE;
    const MAX_NODE_COUNT: u32 = SVO_INDIVIDUAL_MAX_NODE_COUNT;
    const MAX_LEAF_COUNT: u32 = SVO_INDIVIDUAL_MAX_LEAF_COUNT;
    const MAX_DEPTH: u32 = SVO_INDIVIDUAL_MAX_DEPTH;
    const LEAF_VOXEL_COUNT: u32 = SVO_INDIVIDUAL_LEAF_VOXEL_COUNT;
    const BRANCH_FACTOR_EXP: u32 = SVO_INDIVIDUAL_BRANCH_FACTOR_EXP;

    fn new() -> Self {
        Self {
            nodes: vec![SVONode::new(); SVO_INDIVIDUAL_MAX_NODE_COUNT as usize],
            leaves: vec![SVOLeaf::new(); SVO_INDIVIDUAL_MAX_LEAF_COUNT as usize],
            counters: SVOCounters {
                node_count: 1,
                leaf_count: 0,
                free_node_count: 0,
                free_leaf_count: 0,
            },
            free_node_indices: vec![0; SVO_INDIVIDUAL_MAX_NODE_COUNT as usize],
            free_leaf_indices: vec![0; SVO_INDIVIDUAL_MAX_LEAF_COUNT as usize],
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

impl SVOIndividual {
    // Helper functions to manage the SVO
    pub fn allocate_node(&mut self) -> usize {
        let index: usize;
        if self.counters.free_node_count > 0 {
            index = self.free_node_indices[(self.counters.free_node_count - 1) as usize] as usize;
            self.counters.free_node_count -= 1;
        } else {
            index = self.counters.node_count as usize;
            self.counters.node_count += 1;
        }

        if index >= Self::MAX_NODE_COUNT as usize {
            panic!("SVO: Out of nodes");
        }

        self.nodes[index] = SVONode::new();

        index
    }
    pub fn allocate_leaf(&mut self) -> usize {
        let index: usize;
        if self.counters.free_leaf_count > 0 {
            index = self.free_leaf_indices[(self.counters.free_leaf_count - 1) as usize] as usize;
            self.counters.free_leaf_count -= 1;
        } else {
            index = self.counters.leaf_count as usize;
            self.counters.leaf_count += 1;
        }

        if index >= Self::MAX_LEAF_COUNT as usize {
            panic!("SVO: Out of leaves");
        }

        self.leaves[index] = SVOLeaf::new();

        index
    }
    pub fn free_node(&mut self, index: usize) {
        if index >= Self::MAX_NODE_COUNT as usize {
            panic!("SVO: Index out of bounds");
        }
        self.free_node_indices[self.counters.free_node_count as usize] = index as u32;
        self.counters.free_node_count += 1;
    }
    pub fn free_leaf(&mut self, index: usize) {
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

        let x_bit =
            shifted_x / (Self::LEAF_VOXEL_COUNT as i32 * (1 << (Self::MAX_DEPTH - depth - 1)));
        let y_bit =
            shifted_y / (Self::LEAF_VOXEL_COUNT as i32 * (1 << (Self::MAX_DEPTH - depth - 1)));
        let z_bit =
            shifted_z / (Self::LEAF_VOXEL_COUNT as i32 * (1 << (Self::MAX_DEPTH - depth - 1)));

        ((x_bit & 1) | ((y_bit & 1) << 1) | ((z_bit & 1) << 2)) as usize
    }

    fn get_leaf_offset(&self, x: i32, y: i32, z: i32) -> usize {
        let shifted_x = x + (WORLD_VOXEL_COUNT as i32 >> 1);
        let shifted_y = y + (WORLD_VOXEL_COUNT as i32 >> 1);
        let shifted_z = z + (WORLD_VOXEL_COUNT as i32 >> 1);

        let x_offset = shifted_x % (Self::LEAF_VOXEL_COUNT as i32);
        let y_offset = shifted_y % (Self::LEAF_VOXEL_COUNT as i32);
        let z_offset = shifted_z % (Self::LEAF_VOXEL_COUNT as i32);

        (x_offset
            + y_offset * (Self::LEAF_VOXEL_COUNT as i32)
            + z_offset * (Self::LEAF_VOXEL_COUNT as i32) * (Self::LEAF_VOXEL_COUNT as i32))
            as usize
    }

    // Set a voxel in the SVO
    fn set_voxel_impl(&mut self, x: i32, y: i32, z: i32, value: bool) {
        let self_ptr = self as *mut Self;
        let mut node_index = ROOT_NODE_INDEX;
        let mut child_index;

        let mut node_indices = [0; Self::MAX_DEPTH as usize];
        let mut child_indices = [0; Self::MAX_DEPTH as usize];

        // Traverse to the appropriate depth
        for depth in 0..Self::MAX_DEPTH - 1 {
            let node = unsafe { &mut (&mut (*self_ptr).nodes)[node_index] };
            child_index = self.get_child_index(x, y, z, depth);
            node_indices[depth as usize] = node_index;
            child_indices[depth as usize] = child_index;

            if !node.has_child(child_index) {
                // if child doesn't exist
                if node.is_filled(child_index) == value {
                    // if child is already set to the same value, return
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
                if node.is_filled(child_index) {
                    // Set all children as filled
                    for i in 0..8 {
                        new_node.set_filled(i, true);
                    }
                } else {
                    // Set all children as not filled
                    for i in 0..8 {
                        new_node.set_filled(i, false);
                    }
                }
            } else {
                node_index = node.get_child_index(child_index) as usize;
            }
        }

        // Handle leaf level
        let node = unsafe { &mut (&mut (*self_ptr).nodes)[node_index] };
        child_index = self.get_child_index(x, y, z, Self::MAX_DEPTH - 1);
        node_indices[Self::MAX_DEPTH as usize - 1] = node_index;
        child_indices[Self::MAX_DEPTH as usize - 1] = child_index;
        if !node.has_child(child_index) {
            // if child doesn't exist, create new leaf
            if node.is_filled(child_index) == value {
                // if child is already set to the same value, return
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
            if node.is_filled(child_index) {
                leaf.voxel_data = u64::MAX;
            } else {
                leaf.voxel_data = 0;
            }
        } else {
            node_index = node.get_child_index(child_index) as usize;
        }

        // Set voxel data in leaf
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
            for j in 0..8 {
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

    // Get a voxel from the SVO
    fn get_voxel_impl(&self, x: i32, y: i32, z: i32) -> bool {
        let self_ptr = self as *const Self;
        let mut node_index = ROOT_NODE_INDEX;
        let mut child_index;

        // Traverse to the appropriate depth
        for depth in 0..Self::MAX_DEPTH - 1 {
            let node = unsafe { &(&(*self_ptr).nodes)[node_index] };
            child_index = self.get_child_index(x, y, z, depth);

            if !node.has_child(child_index) {
                // Child doesn't exist, return filled bit value
                return node.is_filled(child_index);
            }
            node_index = node.get_child_index(child_index) as usize;
        }

        // Handle leaf level
        let node = unsafe { &(&(*self_ptr).nodes)[node_index] };
        child_index = self.get_child_index(x, y, z, Self::MAX_DEPTH - 1);
        if !node.has_child(child_index) {
            // Leaf doesn't exist, return filled bit value
            return node.is_filled(child_index);
        }
        node_index = node.get_child_index(child_index) as usize;

        // Get voxel data from leaf
        let bit_offset = self.get_leaf_offset(x, y, z);
        let leaf = unsafe { &(&(*self_ptr).leaves)[node_index] };
        let mask = 1u64 << bit_offset;

        (leaf.voxel_data & mask) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::svo::{SVO, get_leaf_coords};
    use core::mem::size_of;

    fn node_count_in_use(svo: &SVOIndividual) -> u32 {
        svo.counters().node_count - svo.counters().free_node_count
    }

    fn leaf_count_in_use(svo: &SVOIndividual) -> u32 {
        svo.counters().leaf_count - svo.counters().free_leaf_count
    }

    #[test]
    fn test_get_leaf_coords_uses_svo_constants() {
        let expected = (
            (crate::voxel::svo::WORLD_VOXEL_COUNT / 2) / SVO::LEAF_VOXEL_COUNT,
            (crate::voxel::svo::WORLD_VOXEL_COUNT / 2) / SVO::LEAF_VOXEL_COUNT,
            (crate::voxel::svo::WORLD_VOXEL_COUNT / 2) / SVO::LEAF_VOXEL_COUNT,
        );
        assert_eq!(get_leaf_coords(0, 0, 0), expected);
    }

    #[test]
    fn test_gpu_accessors_match_individual_layout_lengths() {
        let svo = SVOIndividual::new();

        assert_eq!(
            svo.nodes_as_bytes().len(),
            SVO_INDIVIDUAL_MAX_NODE_COUNT as usize * size_of::<SVONode>()
        );
        assert_eq!(
            svo.leaves_as_bytes().len(),
            SVO_INDIVIDUAL_MAX_LEAF_COUNT as usize * size_of::<SVOLeaf>()
        );
        assert_eq!(
            svo.free_node_indices_as_bytes().len(),
            SVO_INDIVIDUAL_MAX_NODE_COUNT as usize * size_of::<u32>()
        );
        assert_eq!(
            svo.free_leaf_indices_as_bytes().len(),
            SVO_INDIVIDUAL_MAX_LEAF_COUNT as usize * size_of::<u32>()
        );
        assert_eq!(svo.counters().node_count, 1);
        assert_eq!(svo.counters().leaf_count, 0);
        assert_eq!(svo.counters().free_node_count, 0);
        assert_eq!(svo.counters().free_leaf_count, 0);
    }

    #[test]
    fn test_not_collapse_when_no_children_and_filled_values_differ() {
        let mut svo = SVOIndividual::new();
        let mut node_indices = [0usize; SVOIndividual::MAX_DEPTH as usize];
        let child_indices = [0usize; SVOIndividual::MAX_DEPTH as usize];

        let mut parent_index = ROOT_NODE_INDEX;
        for depth in 0..SVOIndividual::MAX_DEPTH as usize - 1 {
            let next_node = svo.allocate_node();
            svo.nodes[parent_index].set_child_mask(0, true);
            svo.nodes[parent_index].set_child_index(0, next_node as u32);
            node_indices[depth] = parent_index;
            parent_index = next_node;
        }

        node_indices[SVOIndividual::MAX_DEPTH as usize - 1] = parent_index;
        let leaf_index = svo.allocate_leaf();
        svo.nodes[parent_index].set_child_mask(0, true);
        svo.nodes[parent_index].set_child_index(0, leaf_index as u32);

        svo.nodes[parent_index].set_filled(0, true);
        svo.nodes[parent_index].set_filled(1, true);
        svo.leaves[leaf_index].voxel_data = 0;

        svo.free_nodes_to_be_freed(&node_indices, &child_indices, leaf_index);

        assert_eq!(svo.counters.free_node_count, 0);
        assert!(svo.nodes[ROOT_NODE_INDEX].has_child(0));
    }

    #[test]
    fn test_collapse_fully_filled_leaf_back_into_parent_bit() {
        let mut svo = SVOIndividual::new();

        for z in 0..SVOIndividual::LEAF_VOXEL_COUNT as i32 {
            for y in 0..SVOIndividual::LEAF_VOXEL_COUNT as i32 {
                for x in 0..SVOIndividual::LEAF_VOXEL_COUNT as i32 {
                    svo.set_voxel(x, y, z, true);
                }
            }
        }

        assert!(svo.get_voxel(0, 0, 0));
        assert!(svo.get_voxel(3, 3, 3));
        assert_eq!(leaf_count_in_use(&svo), 0);
        assert_eq!(node_count_in_use(&svo), SVOIndividual::MAX_DEPTH);
    }

    #[test]
    fn test_collapse_intermediate_node_when_eight_leaves_become_full() {
        let mut svo = SVOIndividual::new();
        let leaf_width = SVOIndividual::LEAF_VOXEL_COUNT as i32;

        svo.set_voxel(0, 0, 0, true);
        let node_count_before = node_count_in_use(&svo);

        for z in 0..(2 * leaf_width) {
            for y in 0..(2 * leaf_width) {
                for x in 0..(2 * leaf_width) {
                    svo.set_voxel(x, y, z, true);
                }
            }
        }

        assert!(svo.get_voxel(0, 0, 0));
        assert!(svo.get_voxel(7, 7, 7));
        assert_eq!(leaf_count_in_use(&svo), 0);
        assert!(node_count_in_use(&svo) < node_count_before + 7);
        assert_eq!(node_count_in_use(&svo), SVOIndividual::MAX_DEPTH - 1);
    }

    #[test]
    fn test_collapse_fully_emptied_leaf_back_into_parent_bit() {
        let mut svo = SVOIndividual::new();

        svo.set_voxel(0, 0, 0, true);
        svo.set_voxel(1, 0, 0, true);
        svo.set_voxel(0, 1, 0, true);
        assert_eq!(leaf_count_in_use(&svo), 1);

        let node_count_before = node_count_in_use(&svo);

        svo.set_voxel(0, 0, 0, false);
        svo.set_voxel(1, 0, 0, false);
        svo.set_voxel(0, 1, 0, false);

        assert!(!svo.get_voxel(0, 0, 0));
        assert!(!svo.get_voxel(1, 0, 0));
        assert!(!svo.get_voxel(0, 1, 0));
        assert_eq!(leaf_count_in_use(&svo), 0);
        assert!(node_count_in_use(&svo) < node_count_before);
        assert_eq!(node_count_in_use(&svo), 1);
    }

    #[test]
    fn test_collapse_intermediate_node_when_eight_leaves_become_empty() {
        let mut svo = SVOIndividual::new();
        let leaf_width = SVOIndividual::LEAF_VOXEL_COUNT as i32;

        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    svo.set_voxel(x * leaf_width, y * leaf_width, z * leaf_width, true);
                }
            }
        }

        assert_eq!(leaf_count_in_use(&svo), 8);
        let node_count_with_leaves = node_count_in_use(&svo);

        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    svo.set_voxel(x * leaf_width, y * leaf_width, z * leaf_width, false);
                }
            }
        }

        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    assert!(!svo.get_voxel(x * leaf_width, y * leaf_width, z * leaf_width));
                }
            }
        }

        assert_eq!(leaf_count_in_use(&svo), 0);
        assert!(node_count_in_use(&svo) < node_count_with_leaves);
        assert_eq!(node_count_in_use(&svo), 1);
    }
}
