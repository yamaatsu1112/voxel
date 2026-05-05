use crate::scene::{EmptyScenes, SceneResource, Scenes};
use crate::system::System;
use crate::world::World;

type SceneChecker = Box<dyn Fn(&mut World) -> bool + Send>;
type SceneUpdater = Box<dyn Fn(&mut World) -> Box<dyn Scenes + Send> + Send>;

pub struct Scheduler {
    world: World,
    first_systems: Vec<System>,
    scenes: Box<dyn Scenes + Send>,
    last_systems: Vec<System>,
    systems_once: Vec<System>,
    should_call_on_enter: bool,
    check_scene_changed: SceneChecker,
    update_scenes: SceneUpdater,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            first_systems: Vec::new(),
            scenes: Box::new(EmptyScenes::Empty),
            last_systems: Vec::new(),
            systems_once: Vec::new(),
            should_call_on_enter: false,
            check_scene_changed: Box::new(|_world: &mut World| false),
            update_scenes: Box::new(|_world: &mut World| {
                Box::new(EmptyScenes::Empty) as Box<dyn Scenes + Send>
            }),
        }
    }

    pub fn set_scenes<T: Scenes + Clone + Send + 'static>(&mut self, scenes: T) {
        self.scenes = Box::new(scenes.clone());
        self.world.insert_resource(SceneResource::new(scenes));
        self.should_call_on_enter = true;

        // Set up the scene change checker and updater for this specific type
        self.check_scene_changed = Box::new(|world: &mut World| {
            world
                .get_resource_mut::<SceneResource<T>>()
                .map(|mut res| res.take_scene_changed())
                .unwrap_or(false)
        });

        self.update_scenes = Box::new(|world: &mut World| {
            world
                .get_resource::<SceneResource<T>>()
                .map(|res| Box::new(res.get().clone()) as Box<dyn Scenes + Send>)
                .expect("SceneResource should exist")
        });
    }

    pub fn add_first_system(&mut self, system: System) {
        self.first_systems.push(system);
    }

    pub fn add_last_system(&mut self, system: System) {
        self.last_systems.push(system);
    }

    pub fn add_system_once(&mut self, system: System) {
        self.systems_once.push(system);
    }

    #[allow(dead_code)]
    pub fn insert_resource<R: Send + 'static>(&mut self, resource: R) {
        self.world.insert_resource(resource);
    }

    pub fn update_once(&mut self) {
        for system in self.systems_once.iter() {
            system(&mut self.world);
        }
    }

    pub fn update(&mut self) {
        // Call on_enter_systems when entering a new scene
        if self.should_call_on_enter {
            for system in self.scenes.on_enter_systems().iter() {
                system(&mut self.world);
            }
            self.should_call_on_enter = false;
        }

        for system in self.first_systems.iter() {
            system(&mut self.world);
        }

        for system in self.scenes.on_update_systems().iter() {
            system(&mut self.world);
        }

        // Check if scene was changed and handle transition
        let scene_changed = (self.check_scene_changed)(&mut self.world);

        if scene_changed {
            // Call on_exit_systems for the old scene
            for system in self.scenes.on_exit_systems().iter() {
                system(&mut self.world);
            }

            // Update the internal scenes reference and prepare for on_enter
            self.scenes = (self.update_scenes)(&mut self.world);
            self.should_call_on_enter = true;
        }

        for system in self.last_systems.iter() {
            system(&mut self.world);
        }
    }

    pub fn _get_world(&self) -> &World {
        &self.world
    }

    pub fn get_world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn first_systems(&self) -> &[System] {
        &self.first_systems
    }

    pub fn last_systems(&self) -> &[System] {
        &self.last_systems
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
