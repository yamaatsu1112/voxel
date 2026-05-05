use super::super::backend::{SVOBackend, cast_slice_to_bytes};
use super::super::counters::SVOCounters;
use super::super::{LEAF_VOXEL_COUNT_EXP, WORLD_VOXEL_COUNT, WORLD_VOXEL_COUNT_EXP};
use core::mem::size_of;

const ROOT_NODE_INDEX: usize = 0;
const GROUP_SIZE: usize = 64;
const SV64_GROUPED_MAX_NODE_COUNT: u32 = 65536;
const SV64_GROUPED_MAX_LEAF_COUNT: u32 = 1048576;
const SV64_GROUPED_LEAF_VOXEL_COUNT: u32 = 1 << LEAF_VOXEL_COUNT_EXP;
const SV64_GROUPED_CHILD_BITS_PER_AXIS: u32 = 2;
const SV64_GROUPED_MAX_DEPTH: u32 =
    (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / SV64_GROUPED_CHILD_BITS_PER_AXIS;
const SV64_FILLED_MASK_FULL: [u32; 2] = [u32::MAX, u32::MAX];

#[repr(C)]
#[derive(Copy, Clone)]
struct SV64GroupedNode {
    child_mask: [u32; 2],
    filled_mask: [u32; 2],
    head_pointer: u32,
}

impl SV64GroupedNode {
    fn new() -> Self {
        Self {
            child_mask: [0; 2],
            filled_mask: [0; 2],
            head_pointer: 0,
        }
    }

    fn new_with_fill(filled: bool) -> Self {
        Self {
            child_mask: [0; 2],
            filled_mask: if filled {
                SV64_FILLED_MASK_FULL
            } else {
                [0; 2]
            },
            head_pointer: 0,
        }
    }

    fn bit_word(child_index: usize) -> usize {
        child_index >> 5
    }

    fn bit_mask(child_index: usize) -> u32 {
        1u32 << (child_index & 31)
    }

    fn has_child(&self, child_index: usize) -> bool {
        let word = Self::bit_word(child_index);
        (self.child_mask[word] & Self::bit_mask(child_index)) != 0
    }

    fn is_filled(&self, child_index: usize) -> bool {
        let word = Self::bit_word(child_index);
        (self.filled_mask[word] & Self::bit_mask(child_index)) != 0
    }

    fn get_child_node_index(&self, child_index: usize) -> u32 {
        self.head_pointer + child_index as u32
    }

    fn set_child_mask(&mut self, child_index: usize, value: bool) {
        let word = Self::bit_word(child_index);
        let bit = Self::bit_mask(child_index);
        if value {
            self.child_mask[word] |= bit;
        } else {
            self.child_mask[word] &= !bit;
        }
    }

    fn set_filled(&mut self, child_index: usize, value: bool) {
        let word = Self::bit_word(child_index);
        let bit = Self::bit_mask(child_index);
        if value {
            self.filled_mask[word] |= bit;
        } else {
            self.filled_mask[word] &= !bit;
        }
    }

    fn set_uniform(&mut self, filled: bool) {
        self.child_mask = [0; 2];
        self.filled_mask = if filled {
            SV64_FILLED_MASK_FULL
        } else {
            [0; 2]
        };
        self.head_pointer = 0;
    }

    fn is_all_empty(&self) -> bool {
        self.child_mask == [0; 2] && self.filled_mask == [0; 2]
    }

    fn is_all_full(&self) -> bool {
        self.child_mask == [0; 2] && self.filled_mask == SV64_FILLED_MASK_FULL
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
pub struct SV64Grouped {
    nodes: Vec<SV64GroupedNode>,
    leaves: Vec<SVOLeaf>,
    counters: SVOCounters,
    free_node_indices: Vec<u32>,
    free_leaf_indices: Vec<u32>,
}

pub(crate) const GPU_NODE_BYTE_SIZE: u64 = size_of::<SV64GroupedNode>() as u64;
pub(crate) const GPU_LEAF_BYTE_SIZE: u64 = size_of::<SVOLeaf>() as u64;

impl Default for SV64Grouped {
    fn default() -> Self {
        <Self as SVOBackend>::new()
    }
}

impl SVOBackend for SV64Grouped {
    const GPU_NODE_BYTE_SIZE: u64 = GPU_NODE_BYTE_SIZE;
    const GPU_LEAF_BYTE_SIZE: u64 = GPU_LEAF_BYTE_SIZE;
    const MAX_NODE_COUNT: u32 = SV64_GROUPED_MAX_NODE_COUNT;
    const MAX_LEAF_COUNT: u32 = SV64_GROUPED_MAX_LEAF_COUNT;
    const MAX_DEPTH: u32 = SV64_GROUPED_MAX_DEPTH;
    const LEAF_VOXEL_COUNT: u32 = SV64_GROUPED_LEAF_VOXEL_COUNT;
    const BRANCH_FACTOR_EXP: u32 = SV64_GROUPED_CHILD_BITS_PER_AXIS;

    fn new() -> Self {
        Self {
            nodes: vec![SV64GroupedNode::new(); SV64_GROUPED_MAX_NODE_COUNT as usize],
            leaves: vec![SVOLeaf::new(); SV64_GROUPED_MAX_LEAF_COUNT as usize],
            counters: SVOCounters {
                node_count: 1,
                leaf_count: 1,
                free_node_count: 0,
                free_leaf_count: 0,
            },
            free_node_indices: vec![0; SV64_GROUPED_MAX_NODE_COUNT as usize],
            free_leaf_indices: vec![0; SV64_GROUPED_MAX_LEAF_COUNT as usize],
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

impl SV64Grouped {
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
                SV64GroupedNode::new_with_fill(parent.is_filled(child_index));
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

        let shift = (Self::MAX_DEPTH - depth - 1) * Self::BRANCH_FACTOR_EXP + LEAF_VOXEL_COUNT_EXP;
        let mask = (1u32 << Self::BRANCH_FACTOR_EXP) - 1;

        let x_bits = (shifted_x >> shift) as u32 & mask;
        let y_bits = (shifted_y >> shift) as u32 & mask;
        let z_bits = (shifted_z >> shift) as u32 & mask;

        (x_bits | (y_bits << Self::BRANCH_FACTOR_EXP) | (z_bits << (Self::BRANCH_FACTOR_EXP * 2)))
            as usize
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
    use crate::voxel::svo::get_leaf_coords;
    use core::mem::size_of;

    fn node_blocks_in_use(svo: &SV64Grouped) -> u32 {
        1 + ((svo.counters().node_count - 1) / GROUP_SIZE as u32) - svo.counters().free_node_count
    }

    fn leaf_blocks_in_use(svo: &SV64Grouped) -> u32 {
        1 + ((svo.counters().leaf_count - 1) / GROUP_SIZE as u32) - svo.counters().free_leaf_count
    }

    #[test]
    fn test_child_index_extracts_two_bits_per_axis() {
        let svo = SV64Grouped::new();

        assert_eq!(svo.get_child_index(0, 0, 0, 0), 42);
        assert_eq!(svo.get_child_index(-512, -512, -512, 0), 0);
        assert_eq!(svo.get_child_index(511, 511, 511, 3), 63);
    }

    #[test]
    fn test_leaf_coords_match_world_size() {
        assert_eq!(get_leaf_coords(0, 0, 0), (128, 128, 128));
        assert_eq!(get_leaf_coords(-512, -512, -512), (0, 0, 0));
        assert_eq!(get_leaf_coords(511, 511, 511), (255, 255, 255));
    }

    #[test]
    fn test_set_and_get_voxel_across_edges() {
        let mut svo = SV64Grouped::new();
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
    fn test_gpu_accessors_match_sv64_grouped_layout_lengths() {
        let svo = SV64Grouped::new();

        assert_eq!(
            svo.nodes_as_bytes().len(),
            SV64_GROUPED_MAX_NODE_COUNT as usize * size_of::<SV64GroupedNode>()
        );
        assert_eq!(
            svo.leaves_as_bytes().len(),
            SV64_GROUPED_MAX_LEAF_COUNT as usize * size_of::<SVOLeaf>()
        );
        assert_eq!(
            svo.free_node_indices_as_bytes().len(),
            SV64_GROUPED_MAX_NODE_COUNT as usize * size_of::<u32>()
        );
        assert_eq!(
            svo.free_leaf_indices_as_bytes().len(),
            SV64_GROUPED_MAX_LEAF_COUNT as usize * size_of::<u32>()
        );
    }

    #[test]
    fn test_fill_full_leaf_group_collapses_back_into_parent_bits() {
        let mut svo = SV64Grouped::new();
        let leaf_group_width = (1 << SV64Grouped::BRANCH_FACTOR_EXP) as i32;

        for leaf_z in 0..leaf_group_width {
            for leaf_y in 0..leaf_group_width {
                for leaf_x in 0..leaf_group_width {
                    for z in 0..SV64Grouped::LEAF_VOXEL_COUNT as i32 {
                        for y in 0..SV64Grouped::LEAF_VOXEL_COUNT as i32 {
                            for x in 0..SV64Grouped::LEAF_VOXEL_COUNT as i32 {
                                let world_x = leaf_x * SV64Grouped::LEAF_VOXEL_COUNT as i32 + x;
                                let world_y = leaf_y * SV64Grouped::LEAF_VOXEL_COUNT as i32 + y;
                                let world_z = leaf_z * SV64Grouped::LEAF_VOXEL_COUNT as i32 + z;
                                svo.set_voxel(world_x, world_y, world_z, true);
                            }
                        }
                    }
                }
            }
        }

        assert!(svo.get_voxel(0, 0, 0));
        let max_coord = leaf_group_width * SV64Grouped::LEAF_VOXEL_COUNT as i32 - 1;
        assert!(svo.get_voxel(max_coord, max_coord, max_coord));
        assert_eq!(leaf_blocks_in_use(&svo), 1);
        assert_eq!(node_blocks_in_use(&svo), SV64Grouped::MAX_DEPTH);
    }

    #[test]
    fn test_clearing_full_leaf_group_releases_allocated_blocks() {
        let mut svo = SV64Grouped::new();

        for z in 0..SV64Grouped::LEAF_VOXEL_COUNT as i32 {
            for y in 0..SV64Grouped::LEAF_VOXEL_COUNT as i32 {
                for x in 0..SV64Grouped::LEAF_VOXEL_COUNT as i32 {
                    svo.set_voxel(x, y, z, true);
                    svo.set_voxel(x, y, z, false);
                }
            }
        }

        assert!(!svo.get_voxel(0, 0, 0));
        assert_eq!(node_blocks_in_use(&svo), 1);
        assert_eq!(leaf_blocks_in_use(&svo), 1);
    }
}
