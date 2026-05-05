use crate::ecs::world::World;
use engine_input::InputState;
use engine_input::input_event::InputEvent;

pub fn render_input_system(world: &mut World) {
    let events = {
        let Some(events) = world.get_resource::<Vec<InputEvent>>() else {
            return;
        };
        events.into_inner().clone()
    };

    let Some(mut input_state) = world.get_resource_mut::<InputState>() else {
        return;
    };

    for event in &events {
        input_state.apply_event(event);
    }
    input_state.finalize_frame();
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_input::{Key, MouseButton};

    #[test]
    fn render_input_system_returns_when_events_resource_is_missing() {
        let mut world = World::new();
        world.insert_resource(InputState::new());

        render_input_system(&mut world);

        let input_state = world.get_resource::<InputState>().unwrap();
        assert!(!input_state.is_key_pressed(Key::KeyW));
    }

    #[test]
    fn render_input_system_returns_when_input_state_is_missing() {
        let mut world = World::new();
        world.insert_resource(vec![InputEvent::KeyState {
            key: Key::KeyW,
            pressed: true,
        }]);

        render_input_system(&mut world);

        let events = world.get_resource::<Vec<InputEvent>>().unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn render_input_system_applies_events_and_finalizes_frame() {
        let mut world = World::new();
        world.insert_resource(InputState::new());
        world.insert_resource(vec![
            InputEvent::KeyState {
                key: Key::KeyW,
                pressed: true,
            },
            InputEvent::MouseButton {
                button: MouseButton::MB1,
                pressed: true,
            },
            InputEvent::MousePosition(320.0, 240.0),
            InputEvent::MouseDelta(3.0, -1.5),
            InputEvent::MouseDelta(1.0, 0.5),
        ]);

        render_input_system(&mut world);

        let input_state = world.get_resource::<InputState>().unwrap();
        assert!(input_state.is_key_pressed(Key::KeyW));
        assert!(input_state.is_key_just_pressed(Key::KeyW));
        assert!(input_state.is_mouse_button_pressed(MouseButton::MB1));
        assert!(input_state.is_mouse_button_just_pressed(MouseButton::MB1));
        assert_eq!(input_state.get_mouse_position(), (320.0, 240.0));
        assert_eq!(input_state.get_mouse_delta(), (4.0, -1.0));
    }

    #[test]
    fn render_input_system_handles_empty_events_by_clearing_frame_delta() {
        let mut world = World::new();
        let mut input_state = InputState::new();
        input_state.apply_event(&InputEvent::MouseDelta(8.0, -4.0));
        input_state.finalize_frame();
        world.insert_resource(input_state);
        world.insert_resource(Vec::<InputEvent>::new());

        render_input_system(&mut world);

        let input_state = world.get_resource::<InputState>().unwrap();
        assert_eq!(input_state.get_mouse_delta(), (0.0, 0.0));
    }
}
