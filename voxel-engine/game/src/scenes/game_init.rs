use voxel_engine::{RenderChannel, Scene, SceneResource, World};

use crate::scenes::{GameScene, GameScenes, RenderGameScene, RenderScenes};
use crate::systems::RenderSceneCommand;
use crate::systems::{entity_creation_system, terrain_init_system};

#[derive(Clone, PartialEq)]
pub struct GameInitScene;

impl Scene for GameInitScene {
    fn on_enter_systems(&self) -> &[fn(&mut World)] {
        &[
            terrain_init_system,
            entity_creation_system,
            transition_to_game_system,
        ]
    }

    fn on_update_systems(&self) -> &[fn(&mut World)] {
        &[]
    }

    fn on_exit_systems(&self) -> &[fn(&mut World)] {
        &[]
    }
}

fn transition_to_game_system(world: &mut World) {
    world
        .get_resource::<RenderChannel>()
        .unwrap()
        .write::<RenderSceneCommand, _>(|value| {
            *value = Some(RenderSceneCommand(RenderScenes::Game(RenderGameScene)));
        });
    let mut scene_resource = world
        .get_resource_mut::<SceneResource<GameScenes>>()
        .unwrap();
    scene_resource.set(GameScenes::Game(GameScene));
}
