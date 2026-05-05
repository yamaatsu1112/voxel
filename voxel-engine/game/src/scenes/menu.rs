use voxel_engine::{Scene, World};

#[derive(Clone, PartialEq)]
pub struct MenuScene;

impl Scene for MenuScene {
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
