use voxel_engine::{Scene, World};

use crate::systems::{
    apply_previous_collision_system, gravity_system, jump_system, lifetime_system, movement_system,
    moving_object_system, prepare_collision_input_system, send_camera_render_data,
    terrain_generator_system, velocity_apply_system, voxel_system,
};

#[derive(Clone, PartialEq)]
pub struct GameScene;

impl Scene for GameScene {
    fn on_enter_systems(&self) -> &[fn(&mut World)] {
        &[]
    }

    fn on_update_systems(&self) -> &[fn(&mut World)] {
        &[
            apply_previous_collision_system,
            jump_system,
            gravity_system,
            movement_system,
            velocity_apply_system,
            prepare_collision_input_system,
            moving_object_system,
            lifetime_system,
            terrain_generator_system,
            voxel_system,
            send_camera_render_data,
        ]
    }

    fn on_exit_systems(&self) -> &[fn(&mut World)] {
        &[]
    }
}
