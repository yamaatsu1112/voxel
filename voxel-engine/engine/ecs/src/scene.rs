use crate::system::System;

/// A scene represents a collection of systems that run during different phases
/// of the scene's lifecycle.
///
/// Scenes are used to organize game logic based on the current state of the game.
/// For example, you might have separate scenes for a main menu, gameplay, and pause screen.
///
/// # Lifecycle
///
/// 1. **Enter**: When transitioning into this scene, `on_enter_systems()` are executed once
/// 2. **Update**: While in this scene, `on_update_systems()` are executed every frame
/// 3. **Exit**: When transitioning out of this scene, `on_exit_systems()` are executed once
pub trait Scene {
    /// Returns the systems to execute when entering this scene.
    /// These systems run once during scene transition.
    fn on_enter_systems(&self) -> &[System];

    /// Returns the systems to execute every frame while in this scene.
    /// These systems run repeatedly during the main update loop.
    fn on_update_systems(&self) -> &[System];

    /// Returns the systems to execute when exiting this scene.
    /// These systems run once when transitioning to another scene.
    fn on_exit_systems(&self) -> &[System];
}

pub trait Scenes {
    fn on_enter_systems(&self) -> &[System];
    fn on_update_systems(&self) -> &[System];
    fn on_exit_systems(&self) -> &[System];
}

/// Empty scene used as default before any scene is set.
/// Returns empty slices for all system methods.
pub(crate) enum EmptyScenes {
    Empty,
}

impl Scenes for EmptyScenes {
    fn on_enter_systems(&self) -> &[System] {
        &[]
    }

    fn on_update_systems(&self) -> &[System] {
        &[]
    }

    fn on_exit_systems(&self) -> &[System] {
        &[]
    }
}

/// Resource that allows systems to trigger scene transitions.
/// Systems can use this to change the current scene state.
pub struct SceneResource<T: Scenes> {
    current: T,
    scene_changed: bool,
}

impl<T: Scenes> SceneResource<T> {
    pub(crate) fn new(scenes: T) -> Self {
        Self {
            current: scenes,
            scene_changed: false,
        }
    }

    /// Returns a reference to the current scene.
    pub fn get(&self) -> &T {
        &self.current
    }

    /// Changes the current scene and notifies the scheduler.
    /// This will trigger on_exit_systems for the current scene
    /// and on_enter_systems for the new scene.
    pub fn set(&mut self, new_scene: T) {
        self.current = new_scene;
        self.scene_changed = true;
    }

    /// Returns whether the scene was changed since last check.
    /// This method also resets the flag to false.
    pub(crate) fn take_scene_changed(&mut self) -> bool {
        let changed = self.scene_changed;
        self.scene_changed = false;
        changed
    }
}
