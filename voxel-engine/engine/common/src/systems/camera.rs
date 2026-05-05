use crate::components::camera::Camera;
use crate::components::local_transform::LocalTransform;
use crate::components::transform::Transform;
use crate::ecs::query::QueryExt;
use crate::ecs::world::World;
use crate::rendering::typed_channel::{GameChannel, Received};

#[derive(Clone, Copy, Debug)]
pub struct CameraTransformData {
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub local_transform: LocalTransform,
}

pub fn game_camera_update_system(world: &mut World) {
    if let Some(game_channel) = world.get_resource::<GameChannel>() {
        let game_channel = GameChannel::clone(&game_channel);
        game_channel.inject_received_resources(world);

        let camera_data = world
            .get_resource::<Received<CameraTransformData>>()
            .map(|r| *r.as_ref());
        if let Some(camera_data) = camera_data {
            let mut query = world.query::<(&Camera, &mut Transform, &mut LocalTransform)>();
            for (_, (_, transform, local_transform)) in query.iter_mut() {
                transform.rotation.x = camera_data.rotation_x;
                transform.rotation.y = camera_data.rotation_y;
                local_transform.position.x = camera_data.local_transform.position.x;
                local_transform.position.y = camera_data.local_transform.position.y;
                local_transform.position.z = camera_data.local_transform.position.z;
            }
        }
    }
}
