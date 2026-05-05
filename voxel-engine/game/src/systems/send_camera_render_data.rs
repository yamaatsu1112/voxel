use voxel_engine::{
    ensure_world_link_id, CameraRenderData, CameraRenderItem, QueryExt, RenderChannel, Transform,
    World,
};

use crate::components::Player;

pub fn send_camera_render_data(world: &mut World) {
    let channel = world
        .get_resource::<RenderChannel>()
        .map(|channel| RenderChannel::clone(&channel));
    let Some(channel) = channel else {
        return;
    };

    let query = world.query::<(&Player, &Transform)>();
    let players: Vec<_> = query
        .iter()
        .map(|(entity, (_, transform))| (entity, *transform))
        .collect();

    let mut items = Vec::with_capacity(players.len());
    for (entity, player_transform) in players {
        let world_link_id = ensure_world_link_id(world, entity);
        items.push(CameraRenderItem {
            world_link_id,
            player_transform,
        });
    }

    channel.write::<CameraRenderData, _>(|value| {
        *value = Some(CameraRenderData { items });
    });
}

#[cfg(test)]
mod tests {
    use super::send_camera_render_data;
    use crate::components::Player;
    use voxel_engine::{
        engine_types::Vec3, CameraRenderData, Received, RenderChannel, Transform, World,
        WorldLinkAllocator,
    };

    #[test]
    fn send_camera_render_data_writes_player_transform_to_channel() {
        let mut world = World::new();
        let channel = RenderChannel::new();
        world.insert_resource(channel.clone());
        world.insert_resource(WorldLinkAllocator::default());

        let player = world.create_entity();
        world.add_bundle(
            player,
            (
                Player,
                Transform {
                    position: Vec3::new(4.0, 5.0, 6.0),
                    rotation: Vec3::new(1.0, 2.0, 3.0),
                },
            ),
        );

        send_camera_render_data(&mut world);
        channel.swap();
        channel.inject_received_resources(&mut world);

        let received_resource = world
            .get_resource::<Received<CameraRenderData>>()
            .expect("camera render data should be received");
        let received = received_resource.as_ref();
        assert_eq!(received.items.len(), 1);
        assert_eq!(received.items[0].player_transform.position.x, 4.0);
        assert_eq!(received.items[0].player_transform.position.y, 5.0);
        assert_eq!(received.items[0].player_transform.position.z, 6.0);
    }

    #[test]
    fn send_camera_render_data_writes_empty_set_when_player_missing() {
        let mut world = World::new();
        let channel = RenderChannel::new();
        world.insert_resource(channel.clone());
        world.insert_resource(WorldLinkAllocator::default());

        send_camera_render_data(&mut world);
        channel.swap();
        channel.inject_received_resources(&mut world);

        let received = world
            .get_resource::<Received<CameraRenderData>>()
            .expect("camera render data should be received");
        assert!(received.as_ref().items.is_empty());
    }

    #[test]
    fn send_camera_render_data_returns_when_channel_missing() {
        let mut world = World::new();
        let player = world.create_entity();
        world.add_bundle(player, (Player, Transform::default()));

        send_camera_render_data(&mut world);

        assert!(world.get_resource::<Received<CameraRenderData>>().is_none());
    }
}
