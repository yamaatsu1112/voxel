use crate::voxel::object::{
    DynamicVoxelObjectData, DynamicVoxelObjectId, VoxelObjectData, VoxelObjectId,
};

/// Number of SVO data channels: 1 for occupancy + 8 for ID bit planes
pub const NUM_SVO_DATA: usize = 9;

/// Manager for VoxelObject and DynamicVoxelObject data.
/// This is an ECS resource that handles CPU-side voxel data storage.
pub struct VoxelObjectManager {
    voxel_objects: Vec<VoxelObjectData>,
    dynamic_voxel_objects: Vec<DynamicVoxelObjectData>,
    next_voxel_object_id: u64,
    next_dynamic_voxel_object_id: u64,
}

impl Default for VoxelObjectManager {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelObjectManager {
    /// Create a new VoxelObjectManager.
    pub fn new() -> Self {
        Self {
            voxel_objects: Vec::new(),
            dynamic_voxel_objects: Vec::new(),
            next_voxel_object_id: 0,
            next_dynamic_voxel_object_id: 0,
        }
    }

    // === VoxelObject API ===

    /// Create a new VoxelObject and return its ID.
    pub fn create_voxel_object(
        &mut self,
        x_size: usize,
        y_size: usize,
        z_size: usize,
    ) -> VoxelObjectId {
        let data = VoxelObjectData::new(x_size, y_size, z_size);
        let id = VoxelObjectId::new(self.next_voxel_object_id);
        self.next_voxel_object_id += 1;
        self.voxel_objects.push(data);
        id
    }

    /// Set a voxel at the given local coordinates.
    pub fn set_voxel(&mut self, id: VoxelObjectId, x: usize, y: usize, z: usize, value: bool) {
        self.voxel_objects[id.index()].set_voxel(x, y, z, value);
    }

    /// Get a voxel at the given local coordinates.
    pub fn get_voxel(&self, id: VoxelObjectId, x: usize, y: usize, z: usize) -> bool {
        self.voxel_objects[id.index()].get_voxel(x, y, z)
    }

    /// Set the voxel ID at the given local coordinates.
    /// ID 0 is reserved for air (empty), valid IDs are 1-255.
    pub fn set_voxel_id(&mut self, id: VoxelObjectId, x: usize, y: usize, z: usize, voxel_id: u8) {
        self.voxel_objects[id.index()].set_id(x, y, z, voxel_id);
    }

    /// Get a reference to the VoxelObjectData.
    pub fn get_voxel_object_data(&self, id: VoxelObjectId) -> &VoxelObjectData {
        &self.voxel_objects[id.index()]
    }

    /// Get a mutable reference to the VoxelObjectData.
    pub fn get_voxel_object_data_mut(&mut self, id: VoxelObjectId) -> &mut VoxelObjectData {
        &mut self.voxel_objects[id.index()]
    }

    // === DynamicVoxelObject API ===

    /// Create a new DynamicVoxelObject and return its ID.
    pub fn create_dynamic_voxel_object(
        &mut self,
        x_size: usize,
        y_size: usize,
        z_size: usize,
    ) -> DynamicVoxelObjectId {
        let data = DynamicVoxelObjectData::new(x_size, y_size, z_size);
        let id = DynamicVoxelObjectId::new(self.next_dynamic_voxel_object_id);
        self.next_dynamic_voxel_object_id += 1;
        self.dynamic_voxel_objects.push(data);
        id
    }

    /// Set a voxel in a DynamicVoxelObject at the given local coordinates.
    pub fn set_dynamic_voxel(
        &mut self,
        id: DynamicVoxelObjectId,
        x: usize,
        y: usize,
        z: usize,
        value: bool,
    ) {
        self.dynamic_voxel_objects[id.index()].set_voxel(x, y, z, value);
    }

    /// Set the voxel ID of a voxel in a DynamicVoxelObject at the given local coordinates.
    /// ID 0 is reserved for air (empty), valid IDs are 1-255.
    pub fn set_dynamic_voxel_id(
        &mut self,
        id: DynamicVoxelObjectId,
        x: usize,
        y: usize,
        z: usize,
        voxel_id: u8,
    ) {
        self.dynamic_voxel_objects[id.index()].set_id(x, y, z, voxel_id);
    }

    /// Get a reference to the DynamicVoxelObjectData.
    pub fn get_dynamic_voxel_object_data(
        &self,
        id: DynamicVoxelObjectId,
    ) -> &DynamicVoxelObjectData {
        &self.dynamic_voxel_objects[id.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_voxel_object() {
        let mut manager = VoxelObjectManager::new();
        let id1 = manager.create_voxel_object(4, 4, 4);
        let id2 = manager.create_voxel_object(8, 8, 8);

        assert_ne!(id1, id2);
        assert_eq!(id1.index(), 0);
        assert_eq!(id2.index(), 1);
    }

    #[test]
    fn test_voxel_object_set_get() {
        let mut manager = VoxelObjectManager::new();
        let id = manager.create_voxel_object(4, 4, 4);

        assert!(!manager.get_voxel(id, 1, 2, 3));
        manager.set_voxel(id, 1, 2, 3, true);
        assert!(manager.get_voxel(id, 1, 2, 3));
    }

    #[test]
    fn test_voxel_object_id() {
        let mut manager = VoxelObjectManager::new();
        let id = manager.create_voxel_object(4, 4, 4);

        manager.set_voxel_id(id, 0, 0, 0, 42);
        let data = manager.get_voxel_object_data(id);
        let index = 0;
        assert_eq!(data.id_data[index], 42);
    }

    #[test]
    fn test_create_dynamic_voxel_object() {
        let mut manager = VoxelObjectManager::new();
        let id1 = manager.create_dynamic_voxel_object(2, 2, 2);
        let id2 = manager.create_dynamic_voxel_object(4, 4, 4);

        assert_ne!(id1, id2);
        assert_eq!(id1.index(), 0);
        assert_eq!(id2.index(), 1);
    }

    #[test]
    fn test_dynamic_voxel_object_set() {
        let mut manager = VoxelObjectManager::new();
        let id = manager.create_dynamic_voxel_object(4, 4, 4);

        manager.set_dynamic_voxel(id, 1, 2, 3, true);
        let data = manager.get_dynamic_voxel_object_data(id);
        assert!(data.get_voxel(1, 2, 3));
    }

    #[test]
    fn test_dynamic_voxel_object_set_id() {
        let mut manager = VoxelObjectManager::new();
        let id = manager.create_dynamic_voxel_object(4, 4, 4);

        manager.set_dynamic_voxel_id(id, 0, 0, 0, 42);
        let data = manager.get_dynamic_voxel_object_data(id);
        assert_eq!(data.get_id(0, 0, 0), 42);
    }
}
