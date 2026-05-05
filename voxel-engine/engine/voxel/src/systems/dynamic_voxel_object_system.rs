use crate::builders::configs::buffer_configs::*;
use crate::components::dynamic_voxel_object::DynamicVoxelObject;
use crate::components::transform::Transform;
use crate::ecs::query::QueryExt;
use crate::recordables::DynamicVoxelPipelineMutex;
use crate::rendering::executor::Executor;
use crate::voxel::DynamicVoxelObjectId;
use crate::voxel::NUM_SVO_DATA;
use crate::voxel::VOXEL_SIZE;
use crate::voxel::VoxelObjectManager;
use crate::voxel::command::LeafEditCommand;
use engine_app::engine_types::Vec3;

/// System that processes DynamicVoxelObject components and generates LeafEditCommands.
/// Unlike voxel_object_system, this processes all objects every frame without dirty tracking.
pub fn dynamic_voxel_object_system(world: &mut crate::ecs::world::World) {
    // Collect query results first to avoid borrowing issues
    let dynamic_voxel_objects: Vec<(DynamicVoxelObjectId, Vec3)> = world
        .query::<(&DynamicVoxelObject, &Transform)>()
        .iter()
        .map(|(_, (dvo, t))| (dvo.id, t.position))
        .collect();

    // Phase 1: Generate all commands
    let mut all_commands: Vec<Vec<Vec<LeafEditCommand>>> = Vec::new();

    {
        let manager = world.get_resource::<VoxelObjectManager>().unwrap();

        for (id, position) in dynamic_voxel_objects {
            let voxel_data = manager.get_dynamic_voxel_object_data(id);

            // Convert world position to voxel coordinates
            let origin_x = (position.x / VOXEL_SIZE).round() as i32;
            let origin_y = (position.y / VOXEL_SIZE).round() as i32;
            let origin_z = (position.z / VOXEL_SIZE).round() as i32;

            // Generate commands for this object
            let commands_per_svo = voxel_data.generate_commands(origin_x, origin_y, origin_z);
            all_commands.push(commands_per_svo);
        }
    }

    // Phase 2: Merge commands and upload to GPU
    {
        let mut compute = world.get_resource_mut::<Executor>().unwrap();
        let (merged_commands_per_svo, counts) =
            merge_and_count_dynamic_commands_per_svo(all_commands);

        for (svo_index, merged_commands) in merged_commands_per_svo.iter().enumerate() {
            if merged_commands.is_empty() {
                continue;
            }

            let command_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    merged_commands.as_ptr() as *const u8,
                    std::mem::size_of_val(merged_commands.as_slice()),
                )
            };
            compute
                .upload_to_buffer::<DynamicVoxelPipelineMutex, DynamicLeafEditCommandBuffer, u8>(
                    svo_index,
                    command_bytes,
                    0,
                );
        }
        compute.upload_to_buffer::<DynamicVoxelPipelineMutex, DynamicCommandCountBuffer, u32>(
            0, &counts, 0,
        );
    }
}

fn merge_and_count_dynamic_commands_per_svo(
    all_commands: Vec<Vec<Vec<LeafEditCommand>>>,
) -> (Vec<Vec<LeafEditCommand>>, [u32; NUM_SVO_DATA]) {
    let mut commands_per_svo = vec![Vec::new(); NUM_SVO_DATA];
    for object_commands in all_commands {
        for (svo_index, commands) in object_commands.into_iter().enumerate() {
            commands_per_svo[svo_index].extend(commands);
        }
    }

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
    fn merge_and_count_dynamic_commands_per_svo_counts_merged_commands() {
        let mut all_commands = vec![vec![Vec::new(); NUM_SVO_DATA]];
        all_commands[0][1].push(LeafEditCommand {
            leaf_x: 3,
            leaf_y: 2,
            leaf_z: 1,
            voxel_data_low: 0b0001,
            voxel_data_high: 0b0010,
        });
        all_commands[0][1].push(LeafEditCommand {
            leaf_x: 3,
            leaf_y: 2,
            leaf_z: 1,
            voxel_data_low: 0b1000,
            voxel_data_high: 0b0100,
        });

        let (merged_commands, counts) = merge_and_count_dynamic_commands_per_svo(all_commands);
        assert_eq!(counts[1], 1);
        assert_eq!(merged_commands[1][0].voxel_data_high, 0b0110);
    }
}
