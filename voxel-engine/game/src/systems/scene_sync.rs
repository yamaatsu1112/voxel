use voxel_engine::{Received, SceneResource, World};

use crate::scenes::{GameScenes, RenderScenes};

#[derive(Clone)]
pub struct GameSceneCommand(pub GameScenes);

#[derive(Clone)]
pub struct RenderSceneCommand(pub RenderScenes);

pub fn receive_game_scene_command(world: &mut World) {
    let scene_command = world
        .get_resource::<Received<GameSceneCommand>>()
        .map(|received| received.as_ref().0.clone());
    world.remove_resource::<Received<GameSceneCommand>>();

    let Some(scene_command) = scene_command else {
        return;
    };

    let current_scene = world
        .get_resource::<SceneResource<GameScenes>>()
        .map(|scene_resource| scene_resource.get().clone());
    if current_scene.as_ref() == Some(&scene_command) {
        return;
    }

    world
        .get_resource_mut::<SceneResource<GameScenes>>()
        .unwrap()
        .set(scene_command);
}

pub fn receive_render_scene_command(world: &mut World) {
    let scene_command = world
        .get_resource::<Received<RenderSceneCommand>>()
        .map(|received| received.as_ref().0.clone());
    world.remove_resource::<Received<RenderSceneCommand>>();

    let Some(scene_command) = scene_command else {
        return;
    };

    let current_scene = world
        .get_resource::<SceneResource<RenderScenes>>()
        .map(|scene_resource| scene_resource.get().clone());
    if current_scene.as_ref() == Some(&scene_command) {
        return;
    }

    world
        .get_resource_mut::<SceneResource<RenderScenes>>()
        .unwrap()
        .set(scene_command);
}

#[cfg(test)]
mod tests {
    use super::{
        receive_game_scene_command, receive_render_scene_command, GameSceneCommand,
        RenderSceneCommand,
    };
    use crate::scenes::{
        GameScene, GameScenes, MenuScene, RenderGameScene, RenderMenuScene, RenderScenes,
    };
    use voxel_engine::ecs::scheduler::Scheduler;
    use voxel_engine::{GameChannel, RenderChannel, SceneResource};

    #[test]
    fn receive_game_scene_command_updates_scene_and_clears_message() {
        let mut scheduler = Scheduler::new();
        scheduler.set_scenes(GameScenes::Menu(MenuScene));
        let channel = GameChannel::new();
        channel.write::<GameSceneCommand, _>(|value| {
            *value = Some(GameSceneCommand(GameScenes::Game(GameScene)));
        });
        channel.swap();
        channel.inject_received_resources(scheduler.get_world_mut());

        receive_game_scene_command(scheduler.get_world_mut());

        let world = scheduler.get_world_mut();
        let scene = world
            .get_resource::<SceneResource<GameScenes>>()
            .unwrap()
            .get()
            .clone();
        assert!(matches!(scene, GameScenes::Game(GameScene)));
        assert!(world
            .get_resource::<voxel_engine::Received<GameSceneCommand>>()
            .is_none());
    }

    #[test]
    fn receive_render_scene_command_updates_scene_from_channel() {
        let mut scheduler = Scheduler::new();
        let channel = RenderChannel::new();
        scheduler.set_scenes(RenderScenes::Menu(RenderMenuScene));

        channel.write::<RenderSceneCommand, _>(|value| {
            *value = Some(RenderSceneCommand(RenderScenes::Game(RenderGameScene)));
        });
        channel.swap();
        channel.inject_received_resources(scheduler.get_world_mut());

        receive_render_scene_command(scheduler.get_world_mut());

        let world = scheduler.get_world_mut();
        let scene = world
            .get_resource::<SceneResource<RenderScenes>>()
            .unwrap()
            .get()
            .clone();
        assert!(matches!(scene, RenderScenes::Game(RenderGameScene)));
        assert!(world
            .get_resource::<voxel_engine::Received<RenderSceneCommand>>()
            .is_none());
    }
}
