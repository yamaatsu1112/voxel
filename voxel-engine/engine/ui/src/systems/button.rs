use crate::components::button::Button;
use crate::ecs::query::QueryExt;
use engine_input::{InputState, MouseButton};

pub fn button_input_system(world: &mut crate::ecs::world::World) {
    let mut query = world.query::<&mut Button>();
    let input = world.get_resource::<InputState>().unwrap();

    let mouse_position = input.get_mouse_position();
    let is_mouse_button_just_pressed = input.is_mouse_button_just_pressed(MouseButton::MB1);

    for (_, button) in query.iter_mut() {
        button.is_hovered = button.is_point_inside(mouse_position.0, mouse_position.1);

        // Reset is_pressed every frame, then set it if clicked this frame
        button.is_pressed = false;
        if button.is_hovered && is_mouse_button_just_pressed {
            button.is_pressed = true;
        }
    }
}

pub fn button_callback_system(world: &mut crate::ecs::world::World) {
    use crate::ecs::query::QueryExt;

    let query = world.query::<&Button>();
    let buttons_to_execute: Vec<_> = query
        .iter()
        .filter_map(|(_, button)| {
            if button.is_pressed {
                button.on_click
            } else {
                None
            }
        })
        .collect();

    for system in buttons_to_execute {
        system(world);
    }
}
