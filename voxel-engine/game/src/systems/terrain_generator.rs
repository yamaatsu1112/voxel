use voxel_engine::builders::configs::buffer_configs::compute::MAX_LEAF_EDIT_COMMANDS;
use voxel_engine::voxel::command::LeafEditCommand;
use voxel_engine::voxel::{LEAF_VOXEL_COUNT, WORLD_VOXEL_COUNT};
use voxel_engine::{VoxelCommandBuffer, World};

use crate::terrain::{NoiseTerrainGenerator, TerrainLoadingState};

const SURFACE_DEPTH_LAYERS: i32 = 1;

pub fn terrain_generator_system(world: &mut World) {
    let Some(mut loading_state) = world.get_resource_mut::<TerrainLoadingState>() else {
        return;
    };
    if loading_state.is_finished() {
        return;
    }

    let center_leaf = (WORLD_VOXEL_COUNT / LEAF_VOXEL_COUNT / 2) as i32;
    let ring_columns = ring_columns(loading_state.current_ring());
    let mut processed_columns = 0usize;
    let mut occupancy = Vec::with_capacity(MAX_LEAF_EDIT_COMMANDS);
    let mut id_bitplane = Vec::with_capacity(MAX_LEAF_EDIT_COMMANDS);

    for &(offset_x, offset_z) in ring_columns.iter().skip(loading_state.ring_offset()) {
        let leaf_x = center_leaf + offset_x;
        let leaf_z = center_leaf + offset_z;
        processed_columns += 1;

        if !is_valid_leaf_column(leaf_x, leaf_z) {
            continue;
        }

        let commands = generate_column_commands(loading_state.generator(), leaf_x, leaf_z);
        if commands.is_empty() {
            continue;
        }
        if occupancy.len() + commands.len() > MAX_LEAF_EDIT_COMMANDS {
            processed_columns -= 1;
            break;
        }

        occupancy.extend(commands.iter().copied());
        id_bitplane.extend(commands);
    }

    loading_state.advance_within_ring(processed_columns, ring_columns.len());

    if occupancy.is_empty() {
        return;
    }

    let mut command_buffer = world.get_resource_mut::<VoxelCommandBuffer>().unwrap();
    command_buffer.append_raw_commands(0, &occupancy);
    command_buffer.append_raw_commands(8, &id_bitplane);
}

fn generate_column_commands(
    generator: &NoiseTerrainGenerator,
    leaf_x: i32,
    leaf_z: i32,
) -> Vec<LeafEditCommand> {
    let world_half = WORLD_VOXEL_COUNT as i32 / 2;
    let world_leaf_x = leaf_x * LEAF_VOXEL_COUNT as i32 - world_half;
    let world_leaf_z = leaf_z * LEAF_VOXEL_COUNT as i32 - world_half;
    let column_heights = column_heights(generator, world_leaf_x, world_leaf_z);
    let surface_leaf_y_range = surface_leaf_y_range(&column_heights);
    let mut commands = Vec::new();

    for surface_leaf_y in surface_leaf_y_range.0..=surface_leaf_y_range.1 {
        for depth in 0..=SURFACE_DEPTH_LAYERS {
            let world_leaf_y = (surface_leaf_y - depth) * LEAF_VOXEL_COUNT as i32;
            let leaf_y = (world_leaf_y + world_half) / LEAF_VOXEL_COUNT as i32;
            if !(0..WORLD_VOXEL_COUNT as i32 / LEAF_VOXEL_COUNT as i32).contains(&leaf_y) {
                continue;
            }

            if let Some((voxel_data_low, voxel_data_high)) =
                build_leaf_voxel_data(&column_heights, world_leaf_y)
            {
                commands.push(LeafEditCommand {
                    leaf_x: leaf_x as u32,
                    leaf_y: leaf_y as u32,
                    leaf_z: leaf_z as u32,
                    voxel_data_low,
                    voxel_data_high,
                });
            }
        }
    }

    commands.sort_by_key(LeafEditCommand::morton_key);
    commands.dedup_by_key(|command| command.leaf_y);
    commands
}

fn column_heights(
    generator: &NoiseTerrainGenerator,
    world_leaf_x: i32,
    world_leaf_z: i32,
) -> [i32; (LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT) as usize] {
    let mut heights = [0i32; (LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT) as usize];

    for local_z in 0..LEAF_VOXEL_COUNT as i32 {
        for local_x in 0..LEAF_VOXEL_COUNT as i32 {
            let world_x = world_leaf_x + local_x;
            let world_z = world_leaf_z + local_z;
            let column_index = (local_z * LEAF_VOXEL_COUNT as i32 + local_x) as usize;
            heights[column_index] = generator.surface_height(world_x, world_z);
        }
    }

    heights
}

fn surface_leaf_y_range(
    column_heights: &[i32; (LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT) as usize],
) -> (i32, i32) {
    let mut min_surface_leaf_y = i32::MAX;
    let mut max_surface_leaf_y = i32::MIN;

    for &surface_y in column_heights {
        let surface_leaf_y = surface_y.div_euclid(LEAF_VOXEL_COUNT as i32);
        min_surface_leaf_y = min_surface_leaf_y.min(surface_leaf_y);
        max_surface_leaf_y = max_surface_leaf_y.max(surface_leaf_y);
    }

    (min_surface_leaf_y, max_surface_leaf_y)
}

fn build_leaf_voxel_data(
    column_heights: &[i32; (LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT) as usize],
    world_leaf_y: i32,
) -> Option<(u32, u32)> {
    let leaf_size = LEAF_VOXEL_COUNT as i32;
    let leaf_top_y = world_leaf_y + leaf_size - 1;
    let mut min_height = i32::MAX;
    let mut max_height = i32::MIN;

    for &column_height in column_heights {
        min_height = min_height.min(column_height);
        max_height = max_height.max(column_height);
    }

    if max_height < world_leaf_y {
        return None;
    }

    if min_height >= leaf_top_y {
        return Some((u32::MAX, u32::MAX));
    }

    let mut voxel_data_low = 0u32;
    let mut voxel_data_high = 0u32;

    for local_z in 0..leaf_size {
        for local_y in 0..leaf_size {
            for local_x in 0..leaf_size {
                let column_index = (local_z * leaf_size + local_x) as usize;
                if world_leaf_y + local_y > column_heights[column_index] {
                    continue;
                }

                let bit_index = local_x as u32 | ((local_y as u32) << 2) | ((local_z as u32) << 4);
                if bit_index < 32 {
                    voxel_data_low |= 1u32 << bit_index;
                } else {
                    voxel_data_high |= 1u32 << (bit_index - 32);
                }
            }
        }
    }

    Some((voxel_data_low, voxel_data_high))
}

fn is_valid_leaf_column(leaf_x: i32, leaf_z: i32) -> bool {
    let leaf_count = WORLD_VOXEL_COUNT as i32 / LEAF_VOXEL_COUNT as i32;
    (0..leaf_count).contains(&leaf_x) && (0..leaf_count).contains(&leaf_z)
}

fn ring_columns(radius: i32) -> Vec<(i32, i32)> {
    let min = -radius - 1;
    let max = radius;
    let side_len = max - min + 1;
    let inner_side_len = side_len - 2;
    let ring_len = if inner_side_len <= 0 {
        side_len * side_len
    } else {
        side_len * side_len - inner_side_len * inner_side_len
    };
    let mut columns = Vec::with_capacity(ring_len as usize);

    for x in min..=max {
        columns.push((x, min));
    }
    for z in (min + 1)..=max {
        columns.push((max, z));
    }
    for x in (min..max).rev() {
        columns.push((x, max));
    }
    for z in ((min + 1)..max).rev() {
        columns.push((min, z));
    }

    columns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_columns_returns_expected_counts() {
        assert_eq!(ring_columns(0).len(), 4);
        assert_eq!(ring_columns(1).len(), 12);
        assert_eq!(ring_columns(2).len(), 20);
    }

    #[test]
    fn ring_columns_starts_with_two_by_two_around_origin() {
        assert_eq!(ring_columns(0), vec![(-1, -1), (0, -1), (0, 0), (-1, 0)]);
    }

    #[test]
    fn build_leaf_voxel_data_skips_empty_leaf() {
        let heights = [0; (LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT) as usize];
        assert_eq!(build_leaf_voxel_data(&heights, 4), None);
    }

    #[test]
    fn build_leaf_voxel_data_marks_full_leaf() {
        let heights = [0; (LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT) as usize];
        assert_eq!(
            build_leaf_voxel_data(&heights, -4),
            Some((u32::MAX, u32::MAX))
        );
    }

    #[test]
    fn surface_leaf_y_range_tracks_column_extremes() {
        let mut heights = [0; (LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT) as usize];
        heights[0] = -9;
        heights[1] = 12;

        assert_eq!(surface_leaf_y_range(&heights), (-3, 3));
    }
}
