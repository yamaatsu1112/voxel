use crate::app::App;
use crate::ecs::scene::Scenes;

use winit::event_loop::{ControlFlow, EventLoop};

pub struct VoxelEngine {
    app: App,
}

impl Default for VoxelEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelEngine {
    pub fn new() -> Self {
        Self { app: App::new() }
    }

    pub fn run(&mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);
        let _ = event_loop.run_app(&mut self.app);
        self.app.join_game_thread();
    }

    pub fn set_scenes<T: Scenes + Clone + Send + 'static>(&mut self, scenes: T) {
        self.app.set_scenes(scenes);
    }

    pub fn add_system_once(&mut self, system: fn(&mut crate::ecs::world::World)) {
        self.app.add_system_once(system);
    }

    pub fn set_render_scenes<T: Scenes + Clone + Send + 'static>(&mut self, scenes: T) {
        self.app.set_render_scenes(scenes);
    }

    pub fn add_render_system_once(&mut self, system: fn(&mut crate::ecs::world::World)) {
        self.app.add_render_system_once(system);
    }

    pub fn add_first_system(&mut self, system: fn(&mut crate::ecs::world::World)) {
        self.app.add_first_system(system);
    }

    pub fn add_last_system(&mut self, system: fn(&mut crate::ecs::world::World)) {
        self.app.add_last_system(system);
    }

    pub fn add_render_first_system(&mut self, system: fn(&mut crate::ecs::world::World)) {
        self.app.add_render_first_system(system);
    }

    pub fn add_render_last_system(&mut self, system: fn(&mut crate::ecs::world::World)) {
        self.app.add_render_last_system(system);
    }
}
