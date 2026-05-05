use voxel_engine::engine_types::Vec3;
use voxel_engine::*;

use crate::components::{CameraRef, CollisionState, Player, TPSCameraSettings, Velocity};
use crate::terrain::TerrainLoadingState;

pub fn entity_creation_system(world: &mut World) {
    let spawn_height = world
        .get_resource::<TerrainLoadingState>()
        .map(|terrain| terrain.spawn_height())
        .unwrap_or(0);
    let spawn_y = (spawn_height as f32 + 10.0) * VOXEL_SIZE;

    // Create camera entity first (so we can reference it from player)
    let camera_entity = world.create_entity();

    // Create player entity (the target the camera follows)
    let player_entity = world.create_entity();

    // Create player DynamicVoxelObject (2x4x2 filled cuboid)
    let player_voxel_id = {
        let mut manager = world.get_resource_mut::<VoxelObjectManager>().unwrap();
        let id = manager.create_dynamic_voxel_object(2, 4, 2);
        for z in 0..2 {
            for y in 0..4 {
                for x in 0..2 {
                    manager.set_dynamic_voxel(id, x, y, z, true);
                }
            }
        }
        id
    };

    world.add_bundle(
        player_entity,
        (
            Player,
            CameraRef {
                entity: camera_entity,
            },
            Transform {
                position: Vec3::new(0.0, spawn_y, 0.0),
                rotation: Vec3::ZERO,
            },
            Velocity::default(),
            Collider {
                half_extent: Vec3::new(VOXEL_SIZE * 1.0, VOXEL_SIZE * 2.0, VOXEL_SIZE * 1.0),
            },
            CollisionState::default(),
            DynamicVoxelObject {
                id: player_voxel_id,
            },
        ),
    );

    // Setup camera entity as child of player
    world.add_bundle(
        camera_entity,
        (
            Camera {
                near: 0.1,
                far: 300.0,
                fov: 90.0,
            },
            TPSCameraSettings {
                distance: 2.0,
                height_offset: 0.5,
                pitch_min: -60.0,
                pitch_max: 60.0,
            },
            Parent::new(player_entity),
            LocalTransform::default(),
            Transform::default(),
        ),
    );
}
