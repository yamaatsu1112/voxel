use voxel_engine::{GameChannel, InputState, Key, SceneResource, Window, World};

use crate::scenes::{GameScenes, PauseScene, RenderPauseScene, RenderScenes};
use crate::systems::GameSceneCommand;

pub fn pause_check_system(world: &mut World) {
    let input_state = world.get_resource::<InputState>().unwrap();
    if input_state.is_key_just_pressed(Key::KeyEscape) {
        let window = world.get_resource::<Window>().unwrap();
        window.disable_cursor_grab();

        let mut scene_resource = world
            .get_resource_mut::<SceneResource<RenderScenes>>()
            .unwrap();
        scene_resource.set(RenderScenes::Pause(RenderPauseScene));
        world
            .get_resource::<GameChannel>()
            .unwrap()
            .write::<GameSceneCommand, _>(|value| {
                *value = Some(GameSceneCommand(GameScenes::Pause(PauseScene)));
            });
    }
}
