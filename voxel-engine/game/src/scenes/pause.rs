use voxel_engine::{Scene, World};

#[derive(Clone, PartialEq)]
pub struct PauseScene;

impl Scene for PauseScene {
    fn on_enter_systems(&self) -> &[fn(&mut World)] {
        &[]
    }

    fn on_update_systems(&self) -> &[fn(&mut World)] {
        &[]
    }

    fn on_exit_systems(&self) -> &[fn(&mut World)] {
        &[]
    }
}
