use crate::voxel::command::LeafEditCommand;
use crate::voxel::manager::NUM_SVO_DATA;
use crate::voxel::svo::{
    LEAF_VOXEL_COUNT, LEAF_VOXEL_COUNT_EXP, WORLD_VOXEL_COUNT, get_leaf_coords,
};
use std::collections::HashMap;

/// ID type for VoxelObject, used to reference voxel data in VoxelObjectManager
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelObjectId(u64);

impl VoxelObjectId {
    pub(crate) fn new(id: u64) -> Self {
        Self(id)
    }

    pub(crate) fn index(&self) -> usize {
        self.0 as usize
    }
}

/// ID type for DynamicVoxelObject, used to reference dynamic voxel data in VoxelObjectManager
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynamicVoxelObjectId(u64);

impl DynamicVoxelObjectId {
    pub(crate) fn new(id: u64) -> Self {
        Self(id)
    }

    pub(crate) fn index(&self) -> usize {
        self.0 as usize
    }
}

/// Internal data for a VoxelObject stored in VoxelObjectManager
#[derive(Clone)]
pub struct VoxelObjectData {
    pub(crate) x_size: usize,
    pub(crate) y_size: usize,
    pub(crate) z_size: usize,
    pub(crate) data: Vec<bool>,
    pub(crate) id_data: Vec<u8>,
}

impl VoxelObjectData {
    /// Create a new VoxelObjectData with the given size.
    pub fn new(x_size: usize, y_size: usize, z_size: usize) -> Self {
        let size = x_size * y_size * z_size;
        Self {
            x_size,
            y_size,
            z_size,
            data: vec![false; size],
            id_data: vec![1; size], // Default to voxel ID 1
        }
    }

    /// Set a voxel at the given local coordinates.
    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, value: bool) {
        if x >= self.x_size || y >= self.y_size || z >= self.z_size {
            panic!("VoxelObjectData: coordinates out of bounds");
        }
        let index = x + y * self.x_size + z * self.x_size * self.y_size;
        self.data[index] = value;
    }

    /// Get a voxel at the given local coordinates.
    pub fn get_voxel(&self, x: usize, y: usize, z: usize) -> bool {
        if x >= self.x_size || y >= self.y_size || z >= self.z_size {
            panic!("VoxelObjectData: coordinates out of bounds");
        }
        let index = x + y * self.x_size + z * self.x_size * self.y_size;
        self.data[index]
    }

    /// Set the voxel ID at the given local coordinates.
    /// ID 0 is reserved for air (empty), valid IDs are 1-255.
    pub fn set_id(&mut self, x: usize, y: usize, z: usize, id: u8) {
        if x >= self.x_size || y >= self.y_size || z >= self.z_size {
            panic!("VoxelObjectData: coordinates out of bounds");
        }
        let index = x + y * self.x_size + z * self.x_size * self.y_size;
        self.id_data[index] = id;
    }

    /// Generate LeafEditCommands for this VoxelObject at the given origin.
    /// Returns 9 command vectors: 1 for occupancy data + 8 for ID bit planes.
    /// Index 0: occupancy data
    /// Index 1-8: ID bit planes (Index 1 = bit 7 MSB, Index 8 = bit 0 LSB)
    /// svo_index = 1 + id_bit
    pub fn generate_commands(
        &self,
        origin_x: i32,
        origin_y: i32,
        origin_z: i32,
    ) -> Vec<Vec<LeafEditCommand>> {
        // Group voxels by their leaf coordinates for each SVOIndividual
        type LeafDataMap = HashMap<(u32, u32, u32), u64>;
        let mut leaf_data_per_svo: Vec<LeafDataMap> = vec![HashMap::new(); NUM_SVO_DATA];

        let x_size = self.x_size;
        let y_size = self.y_size;
        let z_size = self.z_size;

        for z in 0..z_size {
            for y in 0..y_size {
                for x in 0..x_size {
                    let index = x + y * x_size + z * x_size * y_size;
                    if !self.data[index] {
                        continue; // Skip empty voxels
                    }

                    let id = self.id_data[index];

                    // Calculate world voxel coordinates
                    let world_x = origin_x + x as i32;
                    let world_y = origin_y + y as i32;
                    let world_z = origin_z + z as i32;

                    // Calculate leaf coordinates
                    let (leaf_x, leaf_y, leaf_z) = get_leaf_coords(world_x, world_y, world_z);

                    // Calculate voxel offset within the leaf (0-3 for each dimension)
                    // Leaf size is 4x4x4 voxels
                    // Must shift coordinates to match the coordinate system used in get_leaf_coords
                    let shifted_x = world_x + (WORLD_VOXEL_COUNT as i32 >> 1);
                    let shifted_y = world_y + (WORLD_VOXEL_COUNT as i32 >> 1);
                    let shifted_z = world_z + (WORLD_VOXEL_COUNT as i32 >> 1);
                    let leaf_mask = LEAF_VOXEL_COUNT as i32 - 1;
                    let local_x = (shifted_x & leaf_mask) as usize;
                    let local_y = (shifted_y & leaf_mask) as usize;
                    let local_z = (shifted_z & leaf_mask) as usize;

                    // Calculate bit index within the 64-bit leaf data
                    let bit_index = local_x
                        | (local_y << LEAF_VOXEL_COUNT_EXP)
                        | (local_z << (LEAF_VOXEL_COUNT_EXP * 2));
                    let voxel_bit = 1u64 << bit_index;

                    // Set occupancy bit (SVOIndividual index 0)
                    let entry = leaf_data_per_svo[0]
                        .entry((leaf_x, leaf_y, leaf_z))
                        .or_insert(0);
                    *entry |= voxel_bit;

                    // Set ID bits (SVOIndividual indices 1-8)
                    // svo_index = 1 + id_bit, where id_bit 7 = MSB
                    for id_bit in 0..8 {
                        let bit_value = (id >> (7 - id_bit)) & 1;
                        if bit_value != 0 {
                            let svo_idx = 1 + id_bit as usize;
                            let entry = leaf_data_per_svo[svo_idx]
                                .entry((leaf_x, leaf_y, leaf_z))
                                .or_insert(0);
                            *entry |= voxel_bit;
                        }
                    }
                }
            }
        }

        // Convert HashMaps to Vec<Vec<LeafEditCommand>>
        leaf_data_per_svo
            .into_iter()
            .map(|leaf_data| {
                let mut commands: Vec<_> = leaf_data
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
            })
            .collect()
    }
}

/// Data for a DynamicVoxelObject stored in the Renderer.
/// Dynamic objects are expected to change every frame and are always transferred to the GPU.
pub struct DynamicVoxelObjectData {
    pub(crate) x_size: usize,
    pub(crate) y_size: usize,
    pub(crate) z_size: usize,
    pub(crate) data: Vec<bool>,
    pub(crate) id_data: Vec<u8>,
}

impl DynamicVoxelObjectData {
    /// Create a new DynamicVoxelObjectData with the given size.
    pub fn new(x_size: usize, y_size: usize, z_size: usize) -> Self {
        let size = x_size * y_size * z_size;
        Self {
            x_size,
            y_size,
            z_size,
            data: vec![false; size],
            id_data: vec![1; size], // Default to voxel ID 1
        }
    }

    /// Set a voxel at the given local coordinates.
    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, value: bool) {
        if x >= self.x_size || y >= self.y_size || z >= self.z_size {
            panic!("DynamicVoxelObjectData: coordinates out of bounds");
        }
        let index = x + y * self.x_size + z * self.x_size * self.y_size;
        self.data[index] = value;
    }

    /// Get a voxel at the given local coordinates.
    pub fn get_voxel(&self, x: usize, y: usize, z: usize) -> bool {
        if x >= self.x_size || y >= self.y_size || z >= self.z_size {
            panic!("DynamicVoxelObjectData: coordinates out of bounds");
        }
        let index = x + y * self.x_size + z * self.x_size * self.y_size;
        self.data[index]
    }

    /// Set the voxel ID at the given local coordinates.
    /// ID 0 is reserved for air (empty), valid IDs are 1-255.
    pub fn set_id(&mut self, x: usize, y: usize, z: usize, id: u8) {
        if x >= self.x_size || y >= self.y_size || z >= self.z_size {
            panic!("DynamicVoxelObjectData: coordinates out of bounds");
        }
        let index = x + y * self.x_size + z * self.x_size * self.y_size;
        self.id_data[index] = id;
    }

    /// Get the voxel ID at the given local coordinates.
    pub fn get_id(&self, x: usize, y: usize, z: usize) -> u8 {
        if x >= self.x_size || y >= self.y_size || z >= self.z_size {
            panic!("DynamicVoxelObjectData: coordinates out of bounds");
        }
        let index = x + y * self.x_size + z * self.x_size * self.y_size;
        self.id_data[index]
    }

    /// Get the size of this data (x, y, z).
    pub fn size(&self) -> (usize, usize, usize) {
        (self.x_size, self.y_size, self.z_size)
    }

    /// Generate LeafEditCommands for this DynamicVoxelObject at the given origin.
    /// Returns 9 command vectors: 1 for occupancy data + 8 for ID bit planes.
    /// Index 0: occupancy data
    /// Index 1-8: ID bit planes (Index 1 = bit 7 MSB, Index 8 = bit 0 LSB)
    /// svo_index = 1 + id_bit
    pub fn generate_commands(
        &self,
        origin_x: i32,
        origin_y: i32,
        origin_z: i32,
    ) -> Vec<Vec<LeafEditCommand>> {
        type LeafDataMap = HashMap<(u32, u32, u32), u64>;
        let mut leaf_data_per_svo: Vec<LeafDataMap> = vec![HashMap::new(); NUM_SVO_DATA];

        let x_size = self.x_size;
        let y_size = self.y_size;
        let z_size = self.z_size;

        for z in 0..z_size {
            for y in 0..y_size {
                for x in 0..x_size {
                    let index = x + y * x_size + z * x_size * y_size;
                    if !self.data[index] {
                        continue;
                    }

                    let id = self.id_data[index];

                    let world_x = origin_x + x as i32;
                    let world_y = origin_y + y as i32;
                    let world_z = origin_z + z as i32;

                    let (leaf_x, leaf_y, leaf_z) = get_leaf_coords(world_x, world_y, world_z);

                    let shifted_x = world_x + (WORLD_VOXEL_COUNT as i32 >> 1);
                    let shifted_y = world_y + (WORLD_VOXEL_COUNT as i32 >> 1);
                    let shifted_z = world_z + (WORLD_VOXEL_COUNT as i32 >> 1);
                    let leaf_mask = LEAF_VOXEL_COUNT as i32 - 1;
                    let local_x = (shifted_x & leaf_mask) as usize;
                    let local_y = (shifted_y & leaf_mask) as usize;
                    let local_z = (shifted_z & leaf_mask) as usize;

                    let bit_index = local_x
                        | (local_y << LEAF_VOXEL_COUNT_EXP)
                        | (local_z << (LEAF_VOXEL_COUNT_EXP * 2));
                    let voxel_bit = 1u64 << bit_index;

                    // Set occupancy bit (SVOIndividual index 0)
                    let entry = leaf_data_per_svo[0]
                        .entry((leaf_x, leaf_y, leaf_z))
                        .or_insert(0);
                    *entry |= voxel_bit;

                    // Set ID bits (SVOIndividual indices 1-8)
                    // svo_index = 1 + id_bit, where id_bit 7 = MSB
                    for id_bit in 0..8 {
                        let bit_value = (id >> (7 - id_bit)) & 1;
                        if bit_value != 0 {
                            let svo_idx = 1 + id_bit as usize;
                            let entry = leaf_data_per_svo[svo_idx]
                                .entry((leaf_x, leaf_y, leaf_z))
                                .or_insert(0);
                            *entry |= voxel_bit;
                        }
                    }
                }
            }
        }

        leaf_data_per_svo
            .into_iter()
            .map(|leaf_data| {
                let mut commands: Vec<_> = leaf_data
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
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // VoxelObjectData tests

    #[test]
    fn test_voxel_object_data_creation() {
        let data = VoxelObjectData::new(4, 4, 4);
        // All voxels should be empty by default
        assert!(!data.get_voxel(0, 0, 0));
        assert!(!data.get_voxel(3, 3, 3));
    }

    #[test]
    fn test_voxel_object_data_set_get_voxel() {
        let mut data = VoxelObjectData::new(4, 4, 4);
        data.set_voxel(1, 2, 3, true);
        assert!(data.get_voxel(1, 2, 3));
        assert!(!data.get_voxel(0, 0, 0));
    }

    #[test]
    fn test_voxel_object_data_set_id() {
        let mut data = VoxelObjectData::new(4, 4, 4);
        data.set_id(1, 2, 3, 42);
        let index = 1 + 2 * data.x_size + 3 * data.x_size * data.y_size;
        assert_eq!(data.id_data[index], 42);
    }

    #[test]
    #[should_panic(expected = "coordinates out of bounds")]
    fn test_voxel_object_data_out_of_bounds_set() {
        let mut data = VoxelObjectData::new(4, 4, 4);
        data.set_voxel(4, 0, 0, true);
    }

    #[test]
    #[should_panic(expected = "coordinates out of bounds")]
    fn test_voxel_object_data_out_of_bounds_get() {
        let data = VoxelObjectData::new(4, 4, 4);
        data.get_voxel(0, 4, 0);
    }

    // VoxelObjectId tests

    #[test]
    fn test_voxel_object_id() {
        let id1 = VoxelObjectId::new(0);
        let id2 = VoxelObjectId::new(1);
        let id1_copy = VoxelObjectId::new(0);

        assert_eq!(id1, id1_copy);
        assert_ne!(id1, id2);
        assert_eq!(id1.index(), 0);
        assert_eq!(id2.index(), 1);
    }

    #[test]
    fn test_dynamic_voxel_object_id() {
        let id1 = DynamicVoxelObjectId::new(0);
        let id2 = DynamicVoxelObjectId::new(1);
        let id1_copy = DynamicVoxelObjectId::new(0);

        assert_eq!(id1, id1_copy);
        assert_ne!(id1, id2);
        assert_eq!(id1.index(), 0);
        assert_eq!(id2.index(), 1);
    }

    // DynamicVoxelObjectData tests

    #[test]
    fn test_dynamic_voxel_object_data_creation() {
        let data = DynamicVoxelObjectData::new(4, 4, 4);
        assert_eq!(data.size(), (4, 4, 4));
        // All voxels should be empty by default
        assert!(!data.get_voxel(0, 0, 0));
        assert!(!data.get_voxel(3, 3, 3));
        // Default ID should be 1
        assert_eq!(data.get_id(0, 0, 0), 1);
    }

    #[test]
    fn test_dynamic_voxel_object_data_set_get_voxel() {
        let mut data = DynamicVoxelObjectData::new(4, 4, 4);
        data.set_voxel(1, 2, 3, true);
        assert!(data.get_voxel(1, 2, 3));
        assert!(!data.get_voxel(0, 0, 0));
    }

    #[test]
    fn test_dynamic_voxel_object_data_set_get_id() {
        let mut data = DynamicVoxelObjectData::new(4, 4, 4);
        data.set_id(1, 2, 3, 42);
        assert_eq!(data.get_id(1, 2, 3), 42);
        // Other voxels should still have default ID
        assert_eq!(data.get_id(0, 0, 0), 1);
    }

    #[test]
    #[should_panic(expected = "coordinates out of bounds")]
    fn test_dynamic_voxel_object_data_out_of_bounds_set() {
        let mut data = DynamicVoxelObjectData::new(4, 4, 4);
        data.set_voxel(4, 0, 0, true);
    }

    #[test]
    #[should_panic(expected = "coordinates out of bounds")]
    fn test_dynamic_voxel_object_data_out_of_bounds_get() {
        let data = DynamicVoxelObjectData::new(4, 4, 4);
        data.get_voxel(0, 4, 0);
    }

    #[test]
    fn test_dynamic_voxel_object_data_generate_commands_empty() {
        let data = DynamicVoxelObjectData::new(4, 4, 4);
        let commands = data.generate_commands(0, 0, 0);
        // 9 command vectors for SVOIndividual (1 occupancy + 8 ID bit planes)
        assert_eq!(commands.len(), 9);
        // All should be empty for empty voxel object
        for cmd_vec in &commands {
            assert!(cmd_vec.is_empty());
        }
    }

    #[test]
    fn test_dynamic_voxel_object_data_generate_commands_single_voxel() {
        let mut data = DynamicVoxelObjectData::new(4, 4, 4);
        data.set_voxel(0, 0, 0, true);
        data.set_id(0, 0, 0, 1);

        let commands = data.generate_commands(0, 0, 0);
        assert_eq!(commands.len(), 9);
        // Occupancy (index 0) should have one command
        assert!(!commands[0].is_empty());
    }

    #[test]
    fn test_voxel_object_generate_commands_sets_upper_32_bits() {
        let mut data = VoxelObjectData::new(4, 4, 4);
        data.set_voxel(0, 0, 2, true);
        data.set_id(0, 0, 2, 0b1000_0000);

        let commands = data.generate_commands(0, 0, 0);

        assert_eq!(commands[0].len(), 1);
        assert_eq!(commands[0][0].voxel_data_low, 0);
        assert_eq!(commands[0][0].voxel_data_high, 1);
        assert_eq!(commands[1].len(), 1);
        assert_eq!(commands[1][0].voxel_data_high, 1);
        for cmd_vec in &commands[2..] {
            assert!(cmd_vec.is_empty());
        }
    }

    #[test]
    fn voxel_object_generate_commands_returns_morton_order() {
        let mut data = VoxelObjectData::new(9, 9, 9);
        data.set_voxel(8, 0, 0, true);
        data.set_voxel(0, 8, 0, true);
        data.set_voxel(0, 0, 8, true);

        let commands = data.generate_commands(0, 0, 0);
        assert_morton_ordered(&commands[0]);
    }

    #[test]
    fn dynamic_voxel_object_generate_commands_returns_morton_order() {
        let mut data = DynamicVoxelObjectData::new(9, 9, 9);
        data.set_voxel(8, 0, 0, true);
        data.set_voxel(0, 8, 0, true);
        data.set_voxel(0, 0, 8, true);

        let commands = data.generate_commands(0, 0, 0);
        assert_morton_ordered(&commands[0]);
    }

    fn assert_morton_ordered(commands: &[LeafEditCommand]) {
        assert!(
            commands
                .windows(2)
                .all(|pair| { pair[0].morton_key() <= pair[1].morton_key() })
        );
    }
}
