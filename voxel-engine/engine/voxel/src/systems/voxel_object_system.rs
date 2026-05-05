use crate::builders::configs::buffer_configs::*;
use crate::recordables::VoxelPipelineMutex;
use crate::rendering::executor::Executor;
use crate::voxel::NUM_SVO_DATA;
use crate::voxel::VoxelCommandBuffer;
use crate::voxel::VoxelDestroyCommandBuffer;
use crate::voxel::command::LeafEditCommand;
use crate::vulkan::resource_config::BufferMarker;
use crate::vulkan::resource_lifetime::Persistent;

/// System that uploads commands accumulated in `VoxelCommandBuffer`.
pub fn voxel_object_system(world: &mut crate::ecs::world::World) {
    let drained_place_commands = {
        let mut command_buffer = world.get_resource_mut::<VoxelCommandBuffer>().unwrap();
        command_buffer.drain()
    };
    let drained_destroy_commands = {
        let mut command_buffer = world
            .get_resource_mut::<VoxelDestroyCommandBuffer>()
            .unwrap();
        command_buffer.drain()
    };

    {
        let mut compute = world.get_resource_mut::<Executor>().unwrap();
        let (merged_place_commands_per_svo, cpu_place_counts) =
            merge_and_count_commands_per_svo(drained_place_commands);
        let (merged_destroy_commands_per_svo, cpu_destroy_counts) =
            merge_and_count_commands_per_svo(drained_destroy_commands);

        upload_commands_at_offset::<LeafEditCommandBuffer>(
            &mut compute,
            &merged_place_commands_per_svo,
        );
        upload_commands_at_offset::<DestroyLeafEditCommandBuffer>(
            &mut compute,
            &merged_destroy_commands_per_svo,
        );

        compute.upload_to_buffer::<VoxelPipelineMutex, CommandCountBuffer, u32>(
            0,
            &cpu_place_counts,
            0,
        );
        compute.upload_to_buffer::<VoxelPipelineMutex, DestroyCommandCountBuffer, u32>(
            0,
            &cpu_destroy_counts,
            0,
        );
    }
}

fn upload_commands_at_offset<T: BufferMarker<Lifetime = Persistent> + Send>(
    compute: &mut Executor,
    commands_per_svo: &[Vec<LeafEditCommand>],
) {
    for (svo_index, merged_commands) in commands_per_svo.iter().enumerate() {
        if merged_commands.is_empty() {
            continue;
        }

        let command_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                merged_commands.as_ptr() as *const u8,
                std::mem::size_of_val(merged_commands.as_slice()),
            )
        };
        compute.upload_to_buffer::<VoxelPipelineMutex, T, u8>(svo_index, command_bytes, 0);
    }
}

fn merge_and_count_commands_per_svo(
    commands_per_svo: Vec<Vec<LeafEditCommand>>,
) -> (Vec<Vec<LeafEditCommand>>, [u32; NUM_SVO_DATA]) {
    let mut merged_commands_per_svo = vec![Vec::new(); NUM_SVO_DATA];
    let mut counts = [0u32; NUM_SVO_DATA];
    for (svo_index, commands) in commands_per_svo.into_iter().enumerate() {
        let merged_commands = LeafEditCommand::merge_commands(&commands);
        counts[svo_index] = merged_commands.len() as u32;
        merged_commands_per_svo[svo_index] = merged_commands;
    }

    (merged_commands_per_svo, counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::NUM_SVO_DATA;

    #[test]
    fn merge_and_count_commands_per_svo_sets_counts_for_empty_slots() {
        let mut all_commands = vec![Vec::new(); NUM_SVO_DATA];
        all_commands[0].push(LeafEditCommand {
            leaf_x: 1,
            leaf_y: 2,
            leaf_z: 3,
            voxel_data_low: 0b0011,
            voxel_data_high: 0b0001,
        });
        all_commands[0].push(LeafEditCommand {
            leaf_x: 1,
            leaf_y: 2,
            leaf_z: 3,
            voxel_data_low: 0b0100,
            voxel_data_high: 0b0100,
        });
        all_commands[2].push(LeafEditCommand {
            leaf_x: 9,
            leaf_y: 9,
            leaf_z: 9,
            voxel_data_low: 0b1111,
            voxel_data_high: 0b1000,
        });

        let (merged_commands, counts) = merge_and_count_commands_per_svo(all_commands);

        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 0);
        assert_eq!(counts[2], 1);
        assert_eq!(merged_commands[0][0].voxel_data_high, 0b0101);
        assert_eq!(merged_commands[2][0].voxel_data_high, 0b1000);
    }
}
