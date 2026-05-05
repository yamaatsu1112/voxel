use voxel_engine::engine_types::Vec2;
use voxel_engine::{Button, GameChannel, QueryExt, SceneResource, Window, WindowSize, World, UI};

use crate::scenes::{GameInitScene, GameScenes, RenderGameScene, RenderScenes};
use crate::systems::GameSceneCommand;

type ButtonDef = (u32, fn(&mut World), f32, f32, f32);

pub fn menu_setup_system(world: &mut World) {
    let window_size = world.get_resource::<WindowSize>().unwrap();
    let center_x = window_size.width / 2;
    let center_y = window_size.height / 2;

    let button_width = 200;
    let button_height = 50;
    let button_spacing = 20;

    let window = world.get_resource::<Window>().unwrap();
    window.disable_cursor_grab();

    let buttons: [ButtonDef; 2] = [
        (0, start_game_callback, 0.2, 0.6, 0.2),
        (1, exit_game_callback, 0.8, 0.2, 0.2),
    ];

    for (index, callback, r, g, b) in buttons.iter() {
        let button_y = center_y - (button_height + button_spacing) * buttons.len() as u32 / 2
            + (*index) * (button_height + button_spacing);
        let button_x = center_x - button_width / 2;

        let entity = world.create_entity();
        world.add_bundle(
            entity,
            (
                UI {
                    position_x: button_x,
                    position_y: button_y,
                    width: button_width,
                    height: button_height,
                    atlas_offset: Vec2::new(*r, *g),
                    atlas_size: Vec2::new(*b, 1.0),
                },
                Button::new(button_x, button_y, button_width, button_height)
                    .with_on_click(*callback),
            ),
        );
    }
}

fn start_game_callback(world: &mut World) {
    let window = world.get_resource::<Window>().unwrap();
    window.enable_cursor_grab();

    let mut scene_resource = world
        .get_resource_mut::<SceneResource<RenderScenes>>()
        .unwrap();
    scene_resource.set(RenderScenes::Game(RenderGameScene));
    world
        .get_resource::<GameChannel>()
        .unwrap()
        .write::<GameSceneCommand, _>(|value| {
            *value = Some(GameSceneCommand(GameScenes::GameInit(GameInitScene)));
        });
}

fn exit_game_callback(world: &mut World) {
    let window = world.get_resource::<Window>().unwrap();
    window.close();
}

pub fn menu_cleanup_system(world: &mut World) {
    let query = world.query::<&UI>();
    let entities_to_remove: Vec<_> = query.iter().map(|(entity, _)| entity).collect();
    for entity in entities_to_remove {
        world.delete_entity(entity);
    }
}
