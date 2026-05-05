mod backend;
mod counters;
pub(crate) mod grouped;
pub(crate) mod individual;
pub(crate) mod sv64_grouped;
pub(crate) mod sv64_individual;
pub(crate) mod svo_config;
pub(crate) mod svo_variant;

use self::svo_config::CurrentSvo;
use self::svo_variant::SvoVariant;
pub use backend::SVOBackend;
pub use counters::SVOCounters;
pub use grouped::SVOGrouped;
pub use individual::SVOIndividual;
use std::mem::size_of;
pub use sv64_grouped::SV64Grouped;
pub use sv64_individual::SV64Individual;

const VOXEL_SIZE_INV: f32 = 16.0;
pub const VOXEL_SIZE: f32 = 1.0 / VOXEL_SIZE_INV; // Size of one voxel in world units
pub const LEAF_VOXEL_COUNT_EXP: u32 = 2;
pub const WORLD_VOXEL_COUNT_EXP: u32 = 10;
pub const MAX_DEPTH: u32 = SVO::MAX_DEPTH;
pub const LEAF_VOXEL_COUNT: u32 = SVO::LEAF_VOXEL_COUNT;
pub const WORLD_VOXEL_COUNT: u32 = 1 << WORLD_VOXEL_COUNT_EXP;

// Public function to calculate leaf coordinates from world coordinates
pub fn get_leaf_coords(x: i32, y: i32, z: i32) -> (u32, u32, u32) {
    let shifted_x = x + (WORLD_VOXEL_COUNT as i32 >> 1);
    let shifted_y = y + (WORLD_VOXEL_COUNT as i32 >> 1);
    let shifted_z = z + (WORLD_VOXEL_COUNT as i32 >> 1);

    let leaf_x = shifted_x / (SVO::LEAF_VOXEL_COUNT as i32);
    let leaf_y = shifted_y / (SVO::LEAF_VOXEL_COUNT as i32);
    let leaf_z = shifted_z / (SVO::LEAF_VOXEL_COUNT as i32);

    (leaf_x as u32, leaf_y as u32, leaf_z as u32)
}

pub type SVO = <CurrentSvo as SvoVariant>::CpuSvo;

pub const SVO_NODE_GPU_BYTE_SIZE: u64 = SVO::GPU_NODE_BYTE_SIZE;
pub const SVO_LEAF_GPU_BYTE_SIZE: u64 = SVO::GPU_LEAF_BYTE_SIZE;
pub const SVO_GPU_NODES_OFFSET: u64 = 0;
pub const SVO_GPU_LEAVES_OFFSET: u64 =
    SVO_GPU_NODES_OFFSET + SVO_NODE_GPU_BYTE_SIZE * SVO::MAX_NODE_COUNT as u64;
pub const SVO_GPU_NODE_COUNT_OFFSET: u64 =
    SVO_GPU_LEAVES_OFFSET + SVO_LEAF_GPU_BYTE_SIZE * SVO::MAX_LEAF_COUNT as u64;
pub const SVO_GPU_LEAF_COUNT_OFFSET: u64 = SVO_GPU_NODE_COUNT_OFFSET + size_of::<u32>() as u64;
pub const SVO_GPU_FREE_NODE_INDICES_OFFSET: u64 =
    SVO_GPU_LEAF_COUNT_OFFSET + size_of::<u32>() as u64;
pub const SVO_GPU_FREE_LEAF_INDICES_OFFSET: u64 =
    SVO_GPU_FREE_NODE_INDICES_OFFSET + size_of::<u32>() as u64 * SVO::MAX_NODE_COUNT as u64;
pub const SVO_GPU_FREE_NODE_COUNT_OFFSET: u64 =
    SVO_GPU_FREE_LEAF_INDICES_OFFSET + size_of::<u32>() as u64 * SVO::MAX_LEAF_COUNT as u64;
pub const SVO_GPU_FREE_LEAF_COUNT_OFFSET: u64 =
    SVO_GPU_FREE_NODE_COUNT_OFFSET + size_of::<u32>() as u64;
pub const SVO_GPU_BYTE_SIZE: u64 = SVO_GPU_FREE_LEAF_COUNT_OFFSET + size_of::<u32>() as u64;

#[cfg(test)]
mod tests {
    use super::{
        LEAF_VOXEL_COUNT_EXP, SV64Individual, SVOBackend, SVOGrouped, SVOIndividual,
        WORLD_VOXEL_COUNT, WORLD_VOXEL_COUNT_EXP, get_leaf_coords,
    };

    #[test]
    fn world_voxel_count_uses_world_voxel_count_exp() {
        let expected = 1 << WORLD_VOXEL_COUNT_EXP;

        assert_eq!(WORLD_VOXEL_COUNT, expected);
    }

    #[test]
    fn get_leaf_coords_returns_non_negative_leaf_indices() {
        let coords: (u32, u32, u32) = get_leaf_coords(-512, 0, 511);

        assert_eq!(coords, (0, 128, 255));
    }

    #[test]
    fn backend_max_depths_are_derived_from_world_size() {
        assert_eq!(
            SVOIndividual::MAX_DEPTH,
            (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / SVOIndividual::BRANCH_FACTOR_EXP
        );
        assert_eq!(
            SVOGrouped::MAX_DEPTH,
            (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / SVOGrouped::BRANCH_FACTOR_EXP
        );
        assert_eq!(
            SV64Individual::MAX_DEPTH,
            (WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP) / SV64Individual::BRANCH_FACTOR_EXP
        );
    }
}
