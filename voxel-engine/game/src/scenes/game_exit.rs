use voxel_engine::{Scene, World};

use crate::systems::game_exit::{cleanup_game_entities_system, transition_to_menu_system};

#[derive(Clone, PartialEq)]
pub struct GameExitScene;

impl Scene for GameExitScene {
    fn on_enter_systems(&self) -> &[fn(&mut World)] {
        &[transition_to_menu_system]
    }

    fn on_update_systems(&self) -> &[fn(&mut World)] {
        &[]
    }

    fn on_exit_systems(&self) -> &[fn(&mut World)] {
        &[cleanup_game_entities_system]
    }
}
