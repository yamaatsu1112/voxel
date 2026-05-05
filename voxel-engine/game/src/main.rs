mod atlas;
mod components;
pub mod configs;
mod scenes;
mod systems;
mod terrain;
mod utils;
mod voxel_texture;

use scenes::{GameScenes, MenuScene, RenderMenuScene, RenderScenes};
use voxel_engine::VoxelEngine;

fn main() {
    voxel_engine::init_logging();

    let mut engine = VoxelEngine::new();
    voxel_engine::setup(&mut engine);
    engine.add_system_once(atlas::setup_texture_atlas);
    engine.add_system_once(voxel_texture::setup_voxel_3d_texture);
    engine.add_first_system(systems::receive_game_scene_command);
    engine.add_render_first_system(systems::receive_render_scene_command);
    engine.set_scenes(GameScenes::Menu(MenuScene));
    engine.set_render_scenes(RenderScenes::Menu(RenderMenuScene));
    engine.run();
}
