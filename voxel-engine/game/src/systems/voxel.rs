use voxel_engine::engine_types::Vec3;
use voxel_engine::*;

use crate::components::{CameraRef, Player};
use crate::utils::calculate_forward_vector_3d;

const EDIT_SHAPE_SIZE: usize = 2;
const EDIT_CENTER_OFFSET: i32 = 1;
const EDIT_MAX_RANGE: f32 = 16.0;
const PLACE_VOXEL_ID: u8 = 2;
const DESTROY_VOXEL_ID: u8 = 1;

fn create_filled_voxel_object_data(voxel_id: u8) -> VoxelObjectData {
    let mut data = VoxelObjectData::new(EDIT_SHAPE_SIZE, EDIT_SHAPE_SIZE, EDIT_SHAPE_SIZE);
    for z in 0..EDIT_SHAPE_SIZE {
        for y in 0..EDIT_SHAPE_SIZE {
            for x in 0..EDIT_SHAPE_SIZE {
                data.set_voxel(x, y, z, true);
                data.set_id(x, y, z, voxel_id);
            }
        }
    }
    data
}

fn world_to_voxel_coord(world: Vec3) -> (i32, i32, i32) {
    (
        (world.x / VOXEL_SIZE).floor() as i32,
        (world.y / VOXEL_SIZE).floor() as i32,
        (world.z / VOXEL_SIZE).floor() as i32,
    )
}

fn edit_origin_from_center(center: (i32, i32, i32)) -> (i32, i32, i32) {
    (
        center.0 - EDIT_CENTER_OFFSET,
        center.1 - EDIT_CENTER_OFFSET,
        center.2 - EDIT_CENTER_OFFSET,
    )
}

pub fn voxel_system(world: &mut World) {
    let (left_clicked, right_clicked) = {
        let input_state = world.get_resource::<InputState>().unwrap();
        (
            input_state.is_mouse_button_just_pressed(MouseButton::MB1),
            input_state.is_mouse_button_just_pressed(MouseButton::MB2),
        )
    };

    if !left_clicked && !right_clicked {
        return;
    }

    let (camera_transform, raycast_result) = {
        let raycast_data = world.get_resource::<VoxelRaycastData>().unwrap();
        let result = if raycast_data.is_hit() {
            Some((raycast_data.voxel_coord, raycast_data.face))
        } else {
            None
        };

        let player_query = world.query::<(&Player, &CameraRef)>();
        let camera_entity = player_query
            .iter()
            .next()
            .map(|(_, (_, camera_ref))| camera_ref.entity);

        let camera_transform =
            camera_entity.and_then(|entity| world.get_component::<Transform>(entity));

        (camera_transform, result)
    };

    let Some(camera_transform) = camera_transform else {
        return;
    };

    let fallback_voxel = || {
        let forward = calculate_forward_vector_3d(&camera_transform);
        let target_world = camera_transform.position + forward * EDIT_MAX_RANGE;
        world_to_voxel_coord(target_world)
    };

    if left_clicked {
        let center_voxel = if let Some((hit, _)) = raycast_result {
            hit
        } else {
            fallback_voxel()
        };
        let origin = edit_origin_from_center(center_voxel);
        let destroy_data = create_filled_voxel_object_data(DESTROY_VOXEL_ID);
        let mut destroy_buffer = world
            .get_resource_mut::<VoxelDestroyCommandBuffer>()
            .unwrap();
        destroy_buffer.append_from(&destroy_data, origin.0, origin.1, origin.2);
    }

    if right_clicked {
        let center_voxel = if let Some((hit, face)) = raycast_result {
            let offset = face_to_offset(face);
            (hit.0 + offset.0, hit.1 + offset.1, hit.2 + offset.2)
        } else {
            fallback_voxel()
        };
        let origin = edit_origin_from_center(center_voxel);
        let place_data = create_filled_voxel_object_data(PLACE_VOXEL_ID);
        let mut place_buffer = world.get_resource_mut::<VoxelCommandBuffer>().unwrap();
        place_buffer.append_from(&place_data, origin.0, origin.1, origin.2);
    }
}

fn face_to_offset(face: u8) -> (i32, i32, i32) {
    match face {
        0 => (1, 0, 0),
        1 => (-1, 0, 0),
        2 => (0, 1, 0),
        3 => (0, -1, 0),
        4 => (0, 0, 1),
        5 => (0, 0, -1),
        _ => (0, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_voxel_coord_handles_positive_and_negative_values() {
        let positive = world_to_voxel_coord(Vec3::new(VOXEL_SIZE * 1.1, 0.0, VOXEL_SIZE * 31.2));
        let negative =
            world_to_voxel_coord(Vec3::new(-0.001, -VOXEL_SIZE * 1.01, -VOXEL_SIZE * 2.0));

        assert_eq!(positive, (1, 0, 31));
        assert_eq!(negative, (-1, -2, -2));
    }

    #[test]
    fn edit_origin_from_center_applies_half_size_offset() {
        assert_eq!(edit_origin_from_center((10, 20, 30)), (9, 19, 29));
        assert_eq!(edit_origin_from_center((0, 0, 0)), (-1, -1, -1));
    }

    #[test]
    fn create_filled_voxel_object_data_sets_all_voxels() {
        let data = create_filled_voxel_object_data(PLACE_VOXEL_ID);

        for z in 0..EDIT_SHAPE_SIZE {
            for y in 0..EDIT_SHAPE_SIZE {
                for x in 0..EDIT_SHAPE_SIZE {
                    assert!(data.get_voxel(x, y, z));
                }
            }
        }
    }
}
