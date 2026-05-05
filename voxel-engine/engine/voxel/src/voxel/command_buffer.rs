use crate::voxel::command::LeafEditCommand;
use crate::voxel::manager::NUM_SVO_DATA;
use crate::voxel::object::VoxelObjectData;

/// ECS resource that stores voxel edit commands until `voxel_object_system` uploads them.
pub struct VoxelCommandBuffer {
    commands: Vec<Vec<LeafEditCommand>>,
}

/// ECS resource that stores voxel destroy commands until `voxel_object_system` uploads them.
pub struct VoxelDestroyCommandBuffer {
    commands: Vec<Vec<LeafEditCommand>>,
}

impl Default for VoxelCommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelCommandBuffer {
    pub fn new() -> Self {
        Self {
            commands: vec![Vec::new(); NUM_SVO_DATA],
        }
    }

    pub fn append_from(
        &mut self,
        data: &VoxelObjectData,
        origin_x: i32,
        origin_y: i32,
        origin_z: i32,
    ) {
        let generated = data.generate_commands(origin_x, origin_y, origin_z);
        for (svo_index, commands) in generated.into_iter().enumerate() {
            self.commands[svo_index].extend(commands);
        }
    }

    pub fn append_raw_commands(&mut self, svo_index: usize, commands: &[LeafEditCommand]) {
        self.commands[svo_index].extend_from_slice(commands);
    }

    pub fn drain(&mut self) -> Vec<Vec<LeafEditCommand>> {
        let mut drained = Vec::new();
        std::mem::swap(&mut self.commands, &mut drained);
        self.commands = vec![Vec::new(); NUM_SVO_DATA];
        drained
    }
}

impl Default for VoxelDestroyCommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelDestroyCommandBuffer {
    pub fn new() -> Self {
        Self {
            commands: vec![Vec::new(); NUM_SVO_DATA],
        }
    }

    pub fn append_from(
        &mut self,
        data: &VoxelObjectData,
        origin_x: i32,
        origin_y: i32,
        origin_z: i32,
    ) {
        let generated = data.generate_commands(origin_x, origin_y, origin_z);
        for (svo_index, commands) in generated.into_iter().enumerate() {
            self.commands[svo_index].extend(commands);
        }
    }

    pub fn drain(&mut self) -> Vec<Vec<LeafEditCommand>> {
        let mut drained = Vec::new();
        std::mem::swap(&mut self.commands, &mut drained);
        self.commands = vec![Vec::new(); NUM_SVO_DATA];
        drained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_from_accumulates_commands() {
        let mut data = VoxelObjectData::new(4, 4, 4);
        data.set_voxel(0, 0, 0, true);

        let mut buffer = VoxelCommandBuffer::new();
        buffer.append_from(&data, 0, 0, 0);
        buffer.append_from(&data, 0, 0, 0);

        let drained = buffer.drain();
        assert_eq!(drained.len(), NUM_SVO_DATA);
        assert_eq!(drained[0].len(), 2);
    }

    #[test]
    fn drain_clears_internal_storage() {
        let mut data = VoxelObjectData::new(4, 4, 4);
        data.set_voxel(0, 0, 0, true);

        let mut buffer = VoxelCommandBuffer::new();
        buffer.append_from(&data, 0, 0, 0);

        let first = buffer.drain();
        let second = buffer.drain();

        assert!(!first[0].is_empty());
        assert!(second.iter().all(|commands| commands.is_empty()));
    }

    #[test]
    fn append_raw_commands_accumulates_commands() {
        let mut buffer = VoxelCommandBuffer::new();
        let commands = [LeafEditCommand {
            leaf_x: 1,
            leaf_y: 2,
            leaf_z: 3,
            voxel_data_low: 1,
            voxel_data_high: 2,
        }];

        buffer.append_raw_commands(0, &commands);
        buffer.append_raw_commands(8, &commands);

        let drained = buffer.drain();
        assert_eq!(drained[0].len(), 1);
        assert_eq!(drained[8].len(), 1);
    }

    #[test]
    fn destroy_buffer_append_from_accumulates_commands() {
        let mut data = VoxelObjectData::new(4, 4, 4);
        data.set_voxel(0, 0, 0, true);

        let mut buffer = VoxelDestroyCommandBuffer::new();
        buffer.append_from(&data, 0, 0, 0);
        buffer.append_from(&data, 0, 0, 0);

        let drained = buffer.drain();
        assert_eq!(drained.len(), NUM_SVO_DATA);
        assert_eq!(drained[0].len(), 2);
    }

    #[test]
    fn destroy_buffer_drain_clears_internal_storage() {
        let mut data = VoxelObjectData::new(4, 4, 4);
        data.set_voxel(0, 0, 0, true);

        let mut buffer = VoxelDestroyCommandBuffer::new();
        buffer.append_from(&data, 0, 0, 0);

        let first = buffer.drain();
        let second = buffer.drain();

        assert!(!first[0].is_empty());
        assert!(second.iter().all(|commands| commands.is_empty()));
    }
}
