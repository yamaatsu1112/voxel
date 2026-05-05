use std::collections::HashMap;

use crate::voxel::svo::{LEAF_VOXEL_COUNT_EXP, WORLD_VOXEL_COUNT_EXP};

const LEAF_COORD_BITS: u32 = WORLD_VOXEL_COUNT_EXP - LEAF_VOXEL_COUNT_EXP;

// Leaf edit command structure - must match compute.slang
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct LeafEditCommand {
    pub leaf_x: u32,
    pub leaf_y: u32,
    pub leaf_z: u32,
    pub voxel_data_low: u32,
    pub voxel_data_high: u32,
}

impl LeafEditCommand {
    pub fn morton_key(&self) -> u32 {
        let mut key = 0u32;
        for bit in 0..LEAF_COORD_BITS {
            key |= ((self.leaf_x >> bit) & 1) << (bit * 3);
            key |= ((self.leaf_y >> bit) & 1) << (bit * 3 + 1);
            key |= ((self.leaf_z >> bit) & 1) << (bit * 3 + 2);
        }
        key
    }

    // Merge multiple leaf edit commands for the same leaf using bitwise OR
    pub fn merge_commands(commands: &[LeafEditCommand]) -> Vec<LeafEditCommand> {
        let mut merged: HashMap<(u32, u32, u32), u64> = HashMap::new();

        for cmd in commands {
            let key = (cmd.leaf_x, cmd.leaf_y, cmd.leaf_z);
            let entry = merged.entry(key).or_insert(0);
            *entry |= u64::from(cmd.voxel_data_low) | (u64::from(cmd.voxel_data_high) << 32);
        }

        let mut commands: Vec<_> = merged
            .into_iter()
            .map(|((leaf_x, leaf_y, leaf_z), voxel_data)| LeafEditCommand {
                leaf_x,
                leaf_y,
                leaf_z,
                voxel_data_low: voxel_data as u32,
                voxel_data_high: (voxel_data >> 32) as u32,
            })
            .collect();
        commands.sort_by_key(LeafEditCommand::morton_key);
        commands
    }
}

#[cfg(test)]
mod tests {
    use super::LeafEditCommand;

    #[test]
    fn merge_commands_preserves_high_bits() {
        let merged = LeafEditCommand::merge_commands(&[
            LeafEditCommand {
                leaf_x: 1,
                leaf_y: 2,
                leaf_z: 3,
                voxel_data_low: 0,
                voxel_data_high: 0b0010,
            },
            LeafEditCommand {
                leaf_x: 1,
                leaf_y: 2,
                leaf_z: 3,
                voxel_data_low: 0b0100,
                voxel_data_high: 0b1000,
            },
        ]);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].voxel_data_low, 0b0100);
        assert_eq!(merged[0].voxel_data_high, 0b1010);
    }

    #[test]
    fn morton_key_interleaves_leaf_coordinate_bits() {
        assert_eq!(command_at(0b01, 0b10, 0b11).morton_key(), 0b110101);
    }

    #[test]
    fn morton_key_uses_all_leaf_coordinate_bits() {
        assert_eq!(command_at(255, 255, 255).morton_key(), (1 << 24) - 1);
    }

    #[test]
    fn merge_commands_returns_morton_order() {
        let merged = LeafEditCommand::merge_commands(&[
            LeafEditCommand {
                leaf_x: 0,
                leaf_y: 1,
                leaf_z: 0,
                voxel_data_low: 1,
                voxel_data_high: 0,
            },
            LeafEditCommand {
                leaf_x: 1,
                leaf_y: 0,
                leaf_z: 0,
                voxel_data_low: 1,
                voxel_data_high: 0,
            },
            LeafEditCommand {
                leaf_x: 0,
                leaf_y: 0,
                leaf_z: 1,
                voxel_data_low: 1,
                voxel_data_high: 0,
            },
        ]);

        let keys: Vec<_> = merged.iter().map(LeafEditCommand::morton_key).collect();
        assert_eq!(keys, vec![1, 2, 4]);
    }

    fn command_at(leaf_x: u32, leaf_y: u32, leaf_z: u32) -> LeafEditCommand {
        LeafEditCommand {
            leaf_x,
            leaf_y,
            leaf_z,
            voxel_data_low: 0,
            voxel_data_high: 0,
        }
    }
}
