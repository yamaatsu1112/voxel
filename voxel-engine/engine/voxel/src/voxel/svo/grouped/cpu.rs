use super::super::backend::{SVOBackend, cast_slice_to_bytes};
use super::super::counters::SVOCounters;
use super::super::{LEAF_VOXEL_COUNT_EXP, WORLD_VOXEL_COUNT, WORLD_VOXEL_COUNT_EXP};
use core::mem::size_of;

const ROOT_NODE_INDEX: usize = 0;
const GROUP_SIZE: usize = 8;
const FILLED_MASK_BITS: u32 = 0xFF00;
const FILLED_MASK_OFFSET: u32 = 8;
const SVO_GROUPED_MAX_NODE_COUNT: u32 = 131072;
const SVO_GROUPED_MAX_LEAF_COUNT: u32 = 524288;
const SVO_GROUPED_BRANCH_FACTOR_EXP: u32 = 1;
const SVO_GROUPED_MAX_DEPTH: u32 =
    (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / SVO_GROUPED_BRANCH_FACTOR_EXP;
const SVO_GROUPED_LEAF_VOXEL_COUNT: u32 = 1 << LEAF_VOXEL_COUNT_EXP;

#[repr(C)]
#[derive(Copy, Clone)]
struct SVOGroupedNode {
    head_pointer: u32,
    masks: u32,
}

impl SVOGroupedNode {
    fn new() -> Self {
        Self {
            head_pointer: 0,
            masks: 0,
        }
    }

    fn new_with_fill(filled: bool) -> Self {
        Self {
            head_pointer: 0,
            masks: if filled { FILLED_MASK_BITS } else { 0 },
        }
    }

    fn has_child(&self, child_index: usize) -> bool {
        (self.masks & (1u32 << child_index)) != 0
    }

    fn is_filled(&self, child_index: usize) -> bool {
        (self.masks & (1u32 << (child_index as u32 + FILLED_MASK_OFFSET))) != 0
    }

    fn get_child_node_index(&self, child_index: usize) -> u32 {
        self.head_pointer + child_index as u32
    }

    fn set_child_mask(&mut self, child_index: usize, value: bool) {
        let bit = 1u32 << child_index;
        if value {
            self.masks |= bit;
        } else {
            self.masks &= !bit;
        }
    }

    fn set_filled(&mut self, child_index: usize, value: bool) {
        let bit = 1u32 << (child_index as u32 + FILLED_MASK_OFFSET);
        if value {
            self.masks |= bit;
        } else {
            self.masks &= !bit;
        }
    }

    fn set_uniform(&mut self, filled: bool) {
        self.head_pointer = 0;
        self.masks = if filled { FILLED_MASK_BITS } else { 0 };
    }

    fn is_all_empty(&self) -> bool {
        self.masks == 0
    }

    fn is_all_full(&self) -> bool {
        self.masks == FILLED_MASK_BITS
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

    fn clear(&mut self) {
        self.voxel_data = 0;
    }

    fn fill(&mut self) {
        self.voxel_data = u64::MAX;
    }

    fn is_empty(&self) -> bool {
        self.voxel_data == 0
    }

    fn is_full(&self) -> bool {
        self.voxel_data == u64::MAX
    }
}

#[derive(Clone)]
pub struct SVOGrouped {
    nodes: Vec<SVOGroupedNode>,
    leaves: Vec<SVOLeaf>,
    counters: SVOCounters,
    free_node_indices: Vec<u32>,
    free_leaf_indices: Vec<u32>,
}

pub(crate) const GPU_NODE_BYTE_SIZE: u64 = size_of::<SVOGroupedNode>() as u64;
pub(crate) const GPU_LEAF_BYTE_SIZE: u64 = size_of::<SVOLeaf>() as u64;

impl Default for SVOGrouped {
    fn default() -> Self {
        <Self as SVOBackend>::new()
    }
}

impl SVOBackend for SVOGrouped {
    const GPU_NODE_BYTE_SIZE: u64 = GPU_NODE_BYTE_SIZE;
    const GPU_LEAF_BYTE_SIZE: u64 = GPU_LEAF_BYTE_SIZE;
    const MAX_NODE_COUNT: u32 = SVO_GROUPED_MAX_NODE_COUNT;
    const MAX_LEAF_COUNT: u32 = SVO_GROUPED_MAX_LEAF_COUNT;
    const MAX_DEPTH: u32 = SVO_GROUPED_MAX_DEPTH;
    const LEAF_VOXEL_COUNT: u32 = SVO_GROUPED_LEAF_VOXEL_COUNT;
    const BRANCH_FACTOR_EXP: u32 = SVO_GROUPED_BRANCH_FACTOR_EXP;

    fn new() -> Self {
        Self {
            nodes: vec![SVOGroupedNode::new(); SVO_GROUPED_MAX_NODE_COUNT as usize],
            leaves: vec![SVOLeaf::new(); SVO_GROUPED_MAX_LEAF_COUNT as usize],
            counters: SVOCounters {
                node_count: 1,
                leaf_count: 1,
                free_node_count: 0,
                free_leaf_count: 0,
            },
            free_node_indices: vec![0; SVO_GROUPED_MAX_NODE_COUNT as usize],
            free_leaf_indices: vec![0; SVO_GROUPED_MAX_LEAF_COUNT as usize],
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

impl SVOGrouped {
    fn allocate_node_block(&mut self, parent_index: usize) -> usize {
        let block_index = if self.counters.free_node_count > 0 {
            self.counters.free_node_count -= 1;
            self.free_node_indices[self.counters.free_node_count as usize] as usize
        } else {
            let next_index = self.counters.node_count as usize;
            self.counters.node_count += GROUP_SIZE as u32;
            next_index
        };

        if block_index == 0 || block_index + GROUP_SIZE > Self::MAX_NODE_COUNT as usize {
            panic!("SVO: Out of grouped nodes");
        }

        let parent = self.nodes[parent_index];
        for child_index in 0..GROUP_SIZE {
            self.nodes[block_index + child_index] =
                SVOGroupedNode::new_with_fill(parent.is_filled(child_index));
        }
        self.nodes[parent_index].head_pointer = block_index as u32;

        block_index
    }

    fn allocate_leaf_block(&mut self, parent_index: usize) -> usize {
        let block_index = if self.counters.free_leaf_count > 0 {
            self.counters.free_leaf_count -= 1;
            self.free_leaf_indices[self.counters.free_leaf_count as usize] as usize
        } else {
            let next_index = self.counters.leaf_count as usize;
            self.counters.leaf_count += GROUP_SIZE as u32;
            next_index
        };

        if block_index == 0 || block_index + GROUP_SIZE > Self::MAX_LEAF_COUNT as usize {
            panic!("SVO: Out of grouped leaves");
        }

        let parent = self.nodes[parent_index];
        for child_index in 0..GROUP_SIZE {
            let leaf = &mut self.leaves[block_index + child_index];
            if parent.is_filled(child_index) {
                leaf.fill();
            } else {
                leaf.clear();
            }
        }
        self.nodes[parent_index].head_pointer = block_index as u32;

        block_index
    }

    fn free_node_block(&mut self, block_index: usize) {
        if block_index == 0 || block_index + GROUP_SIZE > Self::MAX_NODE_COUNT as usize {
            panic!("SVO: Node block index out of bounds");
        }
        self.free_node_indices[self.counters.free_node_count as usize] = block_index as u32;
        self.counters.free_node_count += 1;
    }

    fn free_leaf_block(&mut self, block_index: usize) {
        if block_index == 0 || block_index + GROUP_SIZE > Self::MAX_LEAF_COUNT as usize {
            panic!("SVO: Leaf block index out of bounds");
        }
        self.free_leaf_indices[self.counters.free_leaf_count as usize] = block_index as u32;
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

    fn collapse_uniform_path(
        &mut self,
        node_indices: &[usize; Self::MAX_DEPTH as usize],
        child_indices: &[usize; Self::MAX_DEPTH as usize],
    ) {
        let leaf_parent_depth = Self::MAX_DEPTH as usize - 1;
        let leaf_parent_index = node_indices[leaf_parent_depth];
        let leaf_block_index = self.nodes[leaf_parent_index].head_pointer as usize;
        if leaf_block_index == 0 {
            return;
        }

        let mut all_empty = true;
        let mut all_full = true;
        for child_index in 0..GROUP_SIZE {
            let leaf = &self.leaves[leaf_block_index + child_index];
            if !leaf.is_empty() {
                all_empty = false;
            }
            if !leaf.is_full() {
                all_full = false;
            }
            if !all_empty && !all_full {
                return;
            }
        }

        let leaf_parent_full = all_full;
        self.free_leaf_block(leaf_block_index);
        self.nodes[leaf_parent_index].set_uniform(leaf_parent_full);

        if leaf_parent_depth == 0 {
            return;
        }

        let mut parent_index = node_indices[leaf_parent_depth - 1];
        let mut child_index = child_indices[leaf_parent_depth - 1];
        self.nodes[parent_index].set_child_mask(child_index, false);
        self.nodes[parent_index].set_filled(child_index, leaf_parent_full);

        for depth in (0..leaf_parent_depth).rev() {
            let current_index = node_indices[depth];
            let current_node = self.nodes[current_index];
            let current_full = if current_node.is_all_full() {
                true
            } else if current_node.is_all_empty() {
                false
            } else {
                break;
            };

            let child_block_index = current_node.head_pointer as usize;
            if child_block_index != 0 {
                self.free_node_block(child_block_index);
                self.nodes[current_index].set_uniform(current_full);
            }

            if depth == 0 {
                break;
            }

            parent_index = node_indices[depth - 1];
            child_index = child_indices[depth - 1];
            self.nodes[parent_index].set_child_mask(child_index, false);
            self.nodes[parent_index].set_filled(child_index, current_full);
        }
    }

    fn set_voxel_impl(&mut self, x: i32, y: i32, z: i32, value: bool) {
        let self_ptr = self as *mut Self;
        let mut node_index = ROOT_NODE_INDEX;
        let mut node_indices = [0; Self::MAX_DEPTH as usize];
        let mut child_indices = [0; Self::MAX_DEPTH as usize];

        for depth in 0..Self::MAX_DEPTH - 1 {
            let child_index = self.get_child_index(x, y, z, depth);
            node_indices[depth as usize] = node_index;
            child_indices[depth as usize] = child_index;

            let node = unsafe { &mut (&mut (*self_ptr).nodes)[node_index] };
            if !node.has_child(child_index) {
                if node.is_filled(child_index) == value {
                    return;
                }
                if node.head_pointer == 0 {
                    unsafe {
                        (*self_ptr).allocate_node_block(node_index);
                    }
                }
                node.set_child_mask(child_index, true);
            }

            node_index = node.get_child_node_index(child_index) as usize;
        }

        let child_index = self.get_child_index(x, y, z, Self::MAX_DEPTH - 1);
        node_indices[Self::MAX_DEPTH as usize - 1] = node_index;
        child_indices[Self::MAX_DEPTH as usize - 1] = child_index;

        let node = unsafe { &mut (&mut (*self_ptr).nodes)[node_index] };
        if !node.has_child(child_index) {
            if node.is_filled(child_index) == value {
                return;
            }
            if node.head_pointer == 0 {
                unsafe {
                    (*self_ptr).allocate_leaf_block(node_index);
                }
            }
            node.set_child_mask(child_index, true);
        }

        let leaf_index = node.get_child_node_index(child_index) as usize;
        let leaf = unsafe { &mut (&mut (*self_ptr).leaves)[leaf_index] };
        let bit_offset = self.get_leaf_offset(x, y, z);
        let mask = 1u64 << bit_offset;

        if value {
            leaf.voxel_data |= mask;
        } else {
            leaf.voxel_data &= !mask;
        }

        self.collapse_uniform_path(&node_indices, &child_indices);
    }

    fn get_voxel_impl(&self, x: i32, y: i32, z: i32) -> bool {
        let mut node_index = ROOT_NODE_INDEX;

        for depth in 0..Self::MAX_DEPTH - 1 {
            let node = &self.nodes[node_index];
            let child_index = self.get_child_index(x, y, z, depth);
            if !node.has_child(child_index) {
                return node.is_filled(child_index);
            }
            node_index = node.get_child_node_index(child_index) as usize;
        }

        let node = &self.nodes[node_index];
        let child_index = self.get_child_index(x, y, z, Self::MAX_DEPTH - 1);
        if !node.has_child(child_index) {
            return node.is_filled(child_index);
        }

        let leaf_index = node.get_child_node_index(child_index) as usize;
        let bit_offset = self.get_leaf_offset(x, y, z);
        let mask = 1u64 << bit_offset;
        let leaf = &self.leaves[leaf_index];

        (leaf.voxel_data & mask) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    fn node_blocks_in_use(svo: &SVOGrouped) -> u32 {
        1 + ((svo.counters().node_count - 1) / GROUP_SIZE as u32) - svo.counters().free_node_count
    }

    fn leaf_blocks_in_use(svo: &SVOGrouped) -> u32 {
        1 + ((svo.counters().leaf_count - 1) / GROUP_SIZE as u32) - svo.counters().free_leaf_count
    }

    #[test]
    fn test_gpu_accessors_match_grouped_layout_lengths() {
        let svo = SVOGrouped::new();

        assert_eq!(
            svo.nodes_as_bytes().len(),
            SVO_GROUPED_MAX_NODE_COUNT as usize * size_of::<SVOGroupedNode>()
        );
        assert_eq!(
            svo.leaves_as_bytes().len(),
            SVO_GROUPED_MAX_LEAF_COUNT as usize * size_of::<SVOLeaf>()
        );
        assert_eq!(
            svo.free_node_indices_as_bytes().len(),
            SVO_GROUPED_MAX_NODE_COUNT as usize * size_of::<u32>()
        );
        assert_eq!(
            svo.free_leaf_indices_as_bytes().len(),
            SVO_GROUPED_MAX_LEAF_COUNT as usize * size_of::<u32>()
        );
        assert_eq!(svo.counters().node_count, 1);
        assert_eq!(svo.counters().leaf_count, 1);
        assert_eq!(svo.counters().free_node_count, 0);
        assert_eq!(svo.counters().free_leaf_count, 0);
    }

    #[test]
    fn test_set_and_clear_single_voxel_roundtrip() {
        let mut svo = SVOGrouped::new();

        assert!(!svo.get_voxel(0, 0, 0));

        svo.set_voxel(0, 0, 0, true);

        assert!(svo.get_voxel(0, 0, 0));
        assert!(!svo.get_voxel(1, 0, 0));
        assert!(node_blocks_in_use(&svo) > 1);
        assert!(leaf_blocks_in_use(&svo) > 1);

        svo.set_voxel(0, 0, 0, false);

        assert!(!svo.get_voxel(0, 0, 0));
        assert_eq!(node_blocks_in_use(&svo), 1);
        assert_eq!(leaf_blocks_in_use(&svo), 1);
    }

    #[test]
    fn test_fill_full_leaf_group_collapses_back_into_parent_bits() {
        let mut svo = SVOGrouped::new();
        let leaf_width = SVOGrouped::LEAF_VOXEL_COUNT as i32;

        for leaf_z in 0..2 {
            for leaf_y in 0..2 {
                for leaf_x in 0..2 {
                    for z in 0..leaf_width {
                        for y in 0..leaf_width {
                            for x in 0..leaf_width {
                                svo.set_voxel(
                                    leaf_x * leaf_width + x,
                                    leaf_y * leaf_width + y,
                                    leaf_z * leaf_width + z,
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }

        assert!(svo.get_voxel(0, 0, 0));
        assert!(svo.get_voxel(2 * leaf_width - 1, 2 * leaf_width - 1, 2 * leaf_width - 1));
        assert_eq!(leaf_blocks_in_use(&svo), 1);
        assert_eq!(node_blocks_in_use(&svo), SVOGrouped::MAX_DEPTH);
    }

    #[test]
    fn test_clearing_full_leaf_group_releases_allocated_blocks() {
        let mut svo = SVOGrouped::new();
        let leaf_width = SVOGrouped::LEAF_VOXEL_COUNT as i32;

        for leaf_z in 0..2 {
            for leaf_y in 0..2 {
                for leaf_x in 0..2 {
                    for z in 0..leaf_width {
                        for y in 0..leaf_width {
                            for x in 0..leaf_width {
                                let world_x = leaf_x * leaf_width + x;
                                let world_y = leaf_y * leaf_width + y;
                                let world_z = leaf_z * leaf_width + z;
                                svo.set_voxel(world_x, world_y, world_z, true);
                                svo.set_voxel(world_x, world_y, world_z, false);
                            }
                        }
                    }
                }
            }
        }

        assert!(!svo.get_voxel(0, 0, 0));
        assert_eq!(node_blocks_in_use(&svo), 1);
        assert_eq!(leaf_blocks_in_use(&svo), 1);
    }
}
