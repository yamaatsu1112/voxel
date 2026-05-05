use crate::components::transform::Transform;
use crate::ecs::component::Component;
use crate::ecs::query::QueryExt;
use crate::ecs::world::World;
use crate::rendering::WorldLinkRegistry;
use crate::rendering::interpolation::{Interpolatable, InterpolationTiming};
use crate::rendering::typed_channel::Received;
use crate::systems::frame_system::{EndOfGameTick, RenderFrameReceiveState};
use crate::systems::render_world_camera::{CameraRenderData, CameraRenderItem, CameraRenderTarget};

use engine_app::game_loop::DEFAULT_GAME_TICK_RATE_HZ;
use engine_macro::Component;
use std::time::Instant;

#[derive(Component, Clone, Copy, Debug)]
struct InterpolatedTransformState {
    previous: Transform,
    current: Transform,
}

impl InterpolatedTransformState {
    fn new(initial: Transform) -> Self {
        Self {
            previous: initial,
            current: initial,
        }
    }
}

pub(crate) fn setup_render_world_interpolation_system(world: &mut World) {
    world.insert_resource(WorldLinkRegistry::default());
    world.insert_resource(InterpolationTiming::new(DEFAULT_GAME_TICK_RATE_HZ));
}

pub(crate) fn interpolate_system(world: &mut World) {
    let alpha = world
        .get_resource::<InterpolationTiming>()
        .map(|timing| timing.alpha(Instant::now()))
        .unwrap_or(1.0);

    let mut query = world.query::<(&mut Transform, &InterpolatedTransformState)>();
    for (_, (transform, interpolation_state)) in query.iter_mut() {
        *transform = interpolation_state
            .previous
            .interpolate(&interpolation_state.current, alpha);
    }
}

pub(crate) fn receive_camera_render_data_system(world: &mut World) {
    let has_new_game_tick = world
        .get_resource::<RenderFrameReceiveState>()
        .map(|state| state.has_new_game_tick)
        .unwrap_or(false);
    if !has_new_game_tick {
        return;
    }

    let tick_id = world
        .get_resource::<Received<EndOfGameTick>>()
        .map(|received| received.as_ref().tick_id);
    let Some(tick_id) = tick_id else {
        return;
    };

    {
        let mut interpolation_timing = world
            .get_resource_mut::<InterpolationTiming>()
            .expect("camera interpolation timing missing");
        interpolation_timing.mark_received(Instant::now());
    }

    let received = world
        .get_resource::<Received<CameraRenderData>>()
        .map(|resource| resource.as_ref().clone())
        .unwrap_or_default();
    world.remove_resource::<Received<CameraRenderData>>();

    for item in received.items {
        apply_camera_render_item(world, tick_id, item);
    }
    remove_stale_camera_render_targets(world, tick_id);
}

fn apply_camera_render_item(world: &mut World, tick_id: u64, item: CameraRenderItem) {
    let existing_render_entity = {
        let mut registry = world
            .get_resource_mut::<WorldLinkRegistry>()
            .expect("world link registry missing");
        registry.touch(item.world_link_id, tick_id)
    };

    if let Some(render_entity) = existing_render_entity {
        let state = world
            .get_component_mut::<InterpolatedTransformState>(render_entity)
            .expect("interpolated transform state missing");
        state.previous = state.current;
        state.current = item.player_transform;
        return;
    }

    let render_entity = world.create_entity();
    world.add_bundle(
        render_entity,
        (
            CameraRenderTarget,
            item.world_link_id,
            item.player_transform,
            InterpolatedTransformState::new(item.player_transform),
        ),
    );

    let mut registry = world
        .get_resource_mut::<WorldLinkRegistry>()
        .expect("world link registry missing");
    registry.insert(item.world_link_id, render_entity, tick_id);
}

fn remove_stale_camera_render_targets(world: &mut World, tick_id: u64) {
    let stale_entries = {
        let registry = world
            .get_resource::<WorldLinkRegistry>()
            .expect("world link registry missing");
        registry.stale_entries(tick_id)
    };

    for (world_link_id, render_entity) in stale_entries {
        world.delete_entity(render_entity);
        let mut registry = world
            .get_resource_mut::<WorldLinkRegistry>()
            .expect("world link registry missing");
        registry.remove(world_link_id);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        InterpolatedTransformState, interpolate_system, receive_camera_render_data_system,
        setup_render_world_interpolation_system,
    };
    use crate::components::transform::Transform;
    use crate::ecs::query::QueryExt;
    use crate::ecs::world::World;
    use crate::rendering::interpolation::Interpolatable;
    use crate::rendering::interpolation::InterpolationTiming;
    use crate::rendering::typed_channel::RenderChannel;
    use crate::systems::frame_system::{EndOfGameTick, RenderFrameReceiveState};
    use crate::systems::render_world_camera::{
        CameraRenderData, CameraRenderItem, CameraRenderTarget, setup_render_camera_system,
    };
    use engine_app::WorldLinkId;
    use engine_app::engine_types::Vec3;
    use std::time::{Duration, Instant};

    #[test]
    fn interpolation_helpers_return_none_without_targets() {
        let mut world = World::new();

        interpolate_system(&mut world);

        let query = world.query::<(&CameraRenderTarget, &Transform)>();
        assert!(query.iter().next().is_none());
    }

    #[test]
    fn transform_default_is_stable_in_empty_world() {
        let transform = Transform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
        };
        assert_eq!(transform.position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn receive_only_advances_previous_and_current_on_new_game_tick() {
        let mut world = World::new();
        let channel = RenderChannel::new();
        setup_render_world_interpolation_system(&mut world);

        write_received_camera_data(
            &mut world,
            &channel,
            1,
            vec![CameraRenderItem {
                world_link_id: WorldLinkId {
                    index: 0,
                    generation: 0,
                },
                player_transform: make_transform(0.0),
            }],
            true,
        );
        receive_camera_render_data_system(&mut world);

        let (_, state) = first_interpolation_state(&mut world);
        assert_eq!(state.previous.position.x, 0.0);
        assert_eq!(state.current.position.x, 0.0);

        write_received_camera_data(
            &mut world,
            &channel,
            2,
            vec![CameraRenderItem {
                world_link_id: WorldLinkId {
                    index: 0,
                    generation: 0,
                },
                player_transform: make_transform(10.0),
            }],
            false,
        );
        receive_camera_render_data_system(&mut world);

        let (_, state) = first_interpolation_state(&mut world);
        assert_eq!(state.previous.position.x, 0.0);
        assert_eq!(state.current.position.x, 0.0);

        write_received_camera_data(
            &mut world,
            &channel,
            2,
            vec![CameraRenderItem {
                world_link_id: WorldLinkId {
                    index: 0,
                    generation: 0,
                },
                player_transform: make_transform(10.0),
            }],
            true,
        );
        receive_camera_render_data_system(&mut world);

        let (_, state) = first_interpolation_state(&mut world);
        assert_eq!(state.previous.position.x, 0.0);
        assert_eq!(state.current.position.x, 10.0);
    }

    #[test]
    fn interpolate_system_writes_midpoint_transform() {
        let mut world = World::new();
        let entity = world.create_entity();
        world.add_bundle(
            entity,
            (
                CameraRenderTarget,
                make_transform(0.0),
                InterpolatedTransformState {
                    previous: make_transform(0.0),
                    current: make_transform(10.0),
                },
            ),
        );

        let mut timing = InterpolationTiming::new(4);
        timing.mark_received(Instant::now() - Duration::from_millis(125));
        world.insert_resource(timing);

        interpolate_system(&mut world);

        let transform = world
            .get_component::<Transform>(entity)
            .expect("interpolated transform missing");
        assert!((transform.position.x - 5.0).abs() < 0.5);
    }

    #[test]
    fn interpolate_system_ignores_non_interpolated_transform_entities() {
        let mut world = World::new();

        let camera_entity = world.create_entity();
        world.add_bundle(camera_entity, (Transform::default(),));

        let entity = world.create_entity();
        world.add_bundle(
            entity,
            (
                make_transform(0.0),
                InterpolatedTransformState {
                    previous: make_transform(0.0),
                    current: make_transform(10.0),
                },
            ),
        );

        let mut timing = InterpolationTiming::new(4);
        timing.mark_received(Instant::now() - Duration::from_millis(125));
        world.insert_resource(timing);

        let mut count = 0;
        let mut query = world.query::<(&mut Transform, &InterpolatedTransformState)>();
        for (_, (transform, interpolation_state)) in query.iter_mut() {
            *transform = interpolation_state
                .previous
                .interpolate(&interpolation_state.current, 0.5);
            count += 1;
        }

        let transform = world
            .get_component::<Transform>(entity)
            .expect("interpolated transform missing");
        assert_eq!(count, 1);
        assert!((transform.position.x - 5.0).abs() < 0.5);
    }

    #[test]
    fn query_with_render_camera_and_received_target_yields_only_interpolated_entities() {
        let mut world = World::new();
        let channel = RenderChannel::new();

        setup_render_world_interpolation_system(&mut world);
        setup_render_camera_system(&mut world);

        write_received_camera_data(
            &mut world,
            &channel,
            1,
            vec![CameraRenderItem {
                world_link_id: WorldLinkId {
                    index: 0,
                    generation: 0,
                },
                player_transform: make_transform(10.0),
            }],
            true,
        );
        receive_camera_render_data_system(&mut world);

        let mut count = 0;
        let mut query = world.query::<(&mut Transform, &InterpolatedTransformState)>();
        for _ in query.iter_mut() {
            count += 1;
        }

        assert_eq!(count, 1);
    }

    fn write_received_camera_data(
        world: &mut World,
        channel: &RenderChannel,
        tick_id: u64,
        items: Vec<CameraRenderItem>,
        has_new_game_tick: bool,
    ) {
        channel.write::<EndOfGameTick, _>(|value| *value = Some(EndOfGameTick { tick_id }));
        channel.write::<CameraRenderData, _>(|value| *value = Some(CameraRenderData { items }));
        channel.swap();
        channel.inject_received_resources(world);
        world.insert_resource(RenderFrameReceiveState { has_new_game_tick });
    }

    fn first_interpolation_state(world: &mut World) -> (Transform, InterpolatedTransformState) {
        let query = world.query::<(&Transform, &InterpolatedTransformState)>();
        let (_, (transform, state)) = query
            .iter()
            .next()
            .expect("interpolated target should exist");
        (
            Transform {
                position: transform.position,
                rotation: transform.rotation,
            },
            InterpolatedTransformState {
                previous: state.previous,
                current: state.current,
            },
        )
    }

    fn make_transform(x: f32) -> Transform {
        Transform {
            position: Vec3::new(x, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
        }
    }
}
