use voxel_engine::RenderChannel;
use voxel_engine::{Camera, QueryExt, SceneResource, Transform, World};

use crate::components::Velocity;
use crate::scenes::{GameScenes, MenuScene, RenderMenuScene, RenderScenes};
use crate::systems::RenderSceneCommand;

pub fn transition_to_menu_system(world: &mut World) {
    world
        .get_resource::<RenderChannel>()
        .unwrap()
        .write::<RenderSceneCommand, _>(|value| {
            *value = Some(RenderSceneCommand(RenderScenes::Menu(RenderMenuScene)));
        });
    let mut scene_resource = world
        .get_resource_mut::<SceneResource<GameScenes>>()
        .unwrap();
    scene_resource.set(GameScenes::Menu(MenuScene));
}

pub fn cleanup_game_entities_system(world: &mut World) {
    let query = world.query::<&Camera>();
    let entities_to_remove: Vec<_> = query.iter().map(|(entity, _)| entity).collect();
    for entity in entities_to_remove {
        world.delete_entity(entity);
    }

    let query = world.query::<&Velocity>();
    let entities_to_remove: Vec<_> = query.iter().map(|(entity, _)| entity).collect();
    for entity in entities_to_remove {
        world.delete_entity(entity);
    }

    let query = world.query::<&Transform>();
    let entities_to_remove: Vec<_> = query.iter().map(|(entity, _)| entity).collect();
    for entity in entities_to_remove {
        world.delete_entity(entity);
    }
}
