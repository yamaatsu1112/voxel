use crate::rendering::executor::Executor;
use crate::rendering::render_world_renderer::RenderWorldRenderer;
use crate::rendering::renderer::Renderer;
use crate::rendering::typed_channel::{GameChannel, Received, RenderChannel};
use std::time::Instant;

use crate::game_loop::DeltaTime;

pub struct EndOfGameTick {
    pub tick_id: u64,
}

#[allow(dead_code)]
pub struct RenderFrameReceiveState {
    pub has_new_game_tick: bool,
}

#[derive(Default)]
struct RenderTickSyncState {
    last_seen_tick_id: Option<u64>,
}

impl RenderTickSyncState {
    fn should_mark_received(&mut self, tick_id: u64) -> bool {
        if self.last_seen_tick_id == Some(tick_id) {
            return false;
        }
        self.last_seen_tick_id = Some(tick_id);
        true
    }
}

#[derive(Default)]
struct GameTickCounter {
    tick_id: u64,
}

struct RenderFrameTiming {
    last_frame_instant: Instant,
}

pub fn frame_begin_system(world: &mut crate::ecs::world::World) {
    let mut compute = world.get_resource_mut::<Executor>().unwrap();
    compute.begin_compute();
}

pub fn frame_end_system(world: &mut crate::ecs::world::World) {
    if let Some(mut renderer) = world.get_resource_mut::<Renderer>() {
        renderer.submit_frame();
    }

    if world.get_resource::<GameTickCounter>().is_none() {
        world.insert_resource(GameTickCounter::default());
    }

    let tick_id = {
        let mut counter = world
            .get_resource_mut::<GameTickCounter>()
            .expect("game tick counter missing");
        counter.tick_id = counter.tick_id.wrapping_add(1);
        counter.tick_id
    };

    if let Some(channel) = world.get_resource::<RenderChannel>() {
        channel.write::<EndOfGameTick, _>(|value| *value = Some(EndOfGameTick { tick_id }));
        channel.swap();
    }
}

pub fn render_frame_begin_system(world: &mut crate::ecs::world::World) {
    let now = Instant::now();
    if world.get_resource::<RenderFrameTiming>().is_none() {
        world.insert_resource(RenderFrameTiming {
            last_frame_instant: now,
        });
    }
    let dt = {
        let mut timing = world
            .get_resource_mut::<RenderFrameTiming>()
            .expect("render frame timing missing");
        let dt = now.duration_since(timing.last_frame_instant).as_secs_f64();
        timing.last_frame_instant = now;
        dt
    };
    world.insert_resource(DeltaTime { dt });

    let channel = {
        let Some(channel) = world.get_resource::<RenderChannel>() else {
            world.insert_resource(RenderFrameReceiveState {
                has_new_game_tick: false,
            });
            return;
        };
        RenderChannel::clone(&channel)
    };

    channel.inject_received_resources(world);

    // Check if new game tick is recieved
    if world.get_resource::<RenderTickSyncState>().is_none() {
        world.insert_resource(RenderTickSyncState::default());
    }
    let latest_tick_id = world
        .get_resource::<Received<EndOfGameTick>>()
        .map(|received| received.as_ref().tick_id);
    let has_new_game_tick = if let Some(tick_id) = latest_tick_id {
        let mut sync_state = world
            .get_resource_mut::<RenderTickSyncState>()
            .expect("render tick sync state missing");
        sync_state.should_mark_received(tick_id)
    } else {
        false
    };

    world.insert_resource(RenderFrameReceiveState { has_new_game_tick });
}

pub fn render_frame_end_system(world: &mut crate::ecs::world::World) {
    let Some(mut renderer) = world.get_resource_mut::<RenderWorldRenderer>() else {
        return;
    };
    renderer.submit_frame();

    if let Some(game_channel) = world.get_resource::<GameChannel>() {
        game_channel.swap();
    }
}

#[cfg(test)]
mod tests {
    use super::render_frame_begin_system;
    use crate::ecs::world::World;
    use crate::game_loop::DeltaTime;

    #[test]
    fn render_frame_begin_system_inserts_delta_time_without_channel() {
        let mut world = World::new();

        render_frame_begin_system(&mut world);

        let dt = world
            .get_resource::<DeltaTime>()
            .expect("delta time should be inserted");
        assert!(dt.dt >= 0.0);
    }

    #[test]
    fn render_frame_begin_system_updates_delta_time_on_consecutive_calls() {
        let mut world = World::new();

        render_frame_begin_system(&mut world);
        std::thread::sleep(std::time::Duration::from_millis(1));
        render_frame_begin_system(&mut world);

        let dt = world
            .get_resource::<DeltaTime>()
            .expect("delta time should be inserted");
        assert!(dt.dt > 0.0);
    }
}
