use crate::ecs::world::World;
use crate::input_event_queue::InputEventQueue;
use engine_input::InputState;

pub fn set_input_state(world: &mut World) {
    world.insert_resource(InputState::new());
}

pub fn input_system(world: &mut World) {
    let events = {
        let Some(queue) = world.get_resource::<InputEventQueue>() else {
            return;
        };
        queue.drain()
    };

    let Some(mut input_state) = world.get_resource_mut::<InputState>() else {
        return;
    };

    for event in &events {
        input_state.apply_event(event);
    }
    input_state.finalize_frame();
}
