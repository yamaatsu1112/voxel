use std::sync::Arc;

use crate::ecs::world::World;
use crate::rendering::double_buffer::DoubleBuffer;
use crate::window::WindowSize;

pub fn set_window_size(world: &mut World) {
    let window_size = {
        let buf = world
            .get_resource::<Arc<DoubleBuffer<Option<WindowSize>>>>()
            .expect("WindowSize DoubleBuffer not found");
        let read = buf.read_buffer();
        read.expect("WindowSize DoubleBuffer has no initial value")
    };
    world.insert_resource(window_size);
}

pub fn update_window_size(world: &mut World) {
    let new_size = {
        let buf = world.get_resource::<Arc<DoubleBuffer<Option<WindowSize>>>>();
        let Some(buf) = buf else {
            return;
        };
        let read = buf.read_buffer();
        *read
    };

    if let Some(new_size) = new_size {
        let mut window_size = world.get_resource_mut::<WindowSize>().unwrap();
        window_size.width = new_size.width;
        window_size.height = new_size.height;
    }
}
