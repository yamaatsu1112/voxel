use voxel_engine::voxel::{LEAF_VOXEL_COUNT, WORLD_VOXEL_COUNT};
use voxel_engine::World;

use crate::terrain::{NoiseTerrainGenerator, TerrainGenerator, TerrainLoadingState};

pub fn terrain_init_system(world: &mut World) {
    if world.get_resource::<TerrainLoadingState>().is_some() {
        return;
    }

    let generator = NoiseTerrainGenerator::new();
    let spawn_height = generator.surface_height(0, 0);
    let terrain_generator: &dyn TerrainGenerator = &generator;
    debug_assert!(terrain_generator.is_solid(0, spawn_height, 0));
    let max_radius = (WORLD_VOXEL_COUNT / LEAF_VOXEL_COUNT / 2) as i32;
    world.insert_resource(TerrainLoadingState::new(
        generator,
        spawn_height,
        max_radius,
    ));
}
