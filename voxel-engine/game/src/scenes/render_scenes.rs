use voxel_engine::{Scene, Scenes, World};

use crate::systems::menu_ui::{menu_cleanup_system, menu_setup_system};
use crate::systems::pause_ui::{pause_cleanup_system, pause_setup_system};
use crate::systems::{
    crosshair_cleanup_system, crosshair_setup_system, crosshair_system, pause_check_system,
    ui_system,
};

#[derive(Clone, PartialEq)]
pub struct RenderMenuScene;

impl Scene for RenderMenuScene {
    fn on_enter_systems(&self) -> &[fn(&mut World)] {
        &[menu_setup_system]
    }

    fn on_update_systems(&self) -> &[fn(&mut World)] {
        &[ui_system]
    }

    fn on_exit_systems(&self) -> &[fn(&mut World)] {
        &[menu_cleanup_system]
    }
}

#[derive(Clone, PartialEq)]
pub struct RenderGameScene;

impl Scene for RenderGameScene {
    fn on_enter_systems(&self) -> &[fn(&mut World)] {
        &[crosshair_setup_system]
    }

    fn on_update_systems(&self) -> &[fn(&mut World)] {
        &[pause_check_system, crosshair_system, ui_system]
    }

    fn on_exit_systems(&self) -> &[fn(&mut World)] {
        &[crosshair_cleanup_system]
    }
}

#[derive(Clone, PartialEq)]
pub struct RenderPauseScene;

impl Scene for RenderPauseScene {
    fn on_enter_systems(&self) -> &[fn(&mut World)] {
        &[pause_setup_system]
    }

    fn on_update_systems(&self) -> &[fn(&mut World)] {
        &[ui_system]
    }

    fn on_exit_systems(&self) -> &[fn(&mut World)] {
        &[pause_cleanup_system]
    }
}

#[derive(Scenes, Clone, PartialEq)]
pub enum RenderScenes {
    Menu(RenderMenuScene),
    Game(RenderGameScene),
    Pause(RenderPauseScene),
}
