use crate::ecs::scheduler::Scheduler;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const DEFAULT_GAME_TICK_RATE_HZ: u32 = 32;

#[derive(Debug, Clone, Copy)]
pub struct DeltaTime {
    pub dt: f64,
}

pub struct GameLoop {
    scheduler: Scheduler,
    tick_duration: Duration,
    max_ticks_per_frame: u32,
}

impl GameLoop {
    pub fn new(scheduler: Scheduler, tick_rate: u32, max_ticks_per_frame: u32) -> Self {
        Self {
            scheduler,
            tick_duration: Duration::from_secs_f64(1.0 / tick_rate as f64),
            max_ticks_per_frame,
        }
    }

    pub fn run(&mut self, stop_flag: Arc<AtomicBool>) {
        let fixed_dt = self.tick_duration.as_secs_f64();
        self.scheduler
            .get_world_mut()
            .insert_resource(DeltaTime { dt: fixed_dt });

        let mut previous_time = Instant::now();
        let mut accumulator = Duration::ZERO;

        while !stop_flag.load(Ordering::Relaxed) {
            let now = Instant::now();
            let elapsed = now - previous_time;
            previous_time = now;

            accumulator += elapsed;

            // Spiral of death prevention: clamp accumulator to upper bound
            let max_accumulator = self.tick_duration * self.max_ticks_per_frame;
            if accumulator > max_accumulator {
                accumulator = max_accumulator;
            }

            while accumulator >= self.tick_duration {
                accumulator -= self.tick_duration;
                self.scheduler.update();
            }

            // Sleep until next tick
            let remaining = self.tick_duration - accumulator;
            if !remaining.is_zero() {
                std::thread::sleep(remaining);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_sets_tick_duration() {
        let scheduler = Scheduler::new();
        let game_loop = GameLoop::new(scheduler, 64, 5);
        let expected = Duration::from_secs_f64(1.0 / 64.0);
        assert_eq!(game_loop.tick_duration, expected);
        assert_eq!(game_loop.max_ticks_per_frame, 5);
    }

    #[test]
    fn test_new_with_different_tick_rates() {
        let scheduler = Scheduler::new();
        let game_loop = GameLoop::new(scheduler, 30, 3);
        let expected = Duration::from_secs_f64(1.0 / 30.0);
        assert_eq!(game_loop.tick_duration, expected);
        assert_eq!(game_loop.max_ticks_per_frame, 3);
    }

    #[test]
    fn test_run_inserts_delta_time_and_stops_immediately() {
        let scheduler = Scheduler::new();
        let mut game_loop = GameLoop::new(scheduler, 64, 5);
        let stop_flag = Arc::new(AtomicBool::new(true)); // Already stopped
        game_loop.run(Arc::clone(&stop_flag));

        let dt = game_loop
            .scheduler
            .get_world_mut()
            .get_resource::<DeltaTime>()
            .expect("DeltaTime should be inserted");
        let expected_dt = 1.0 / 64.0;
        assert!((dt.dt - expected_dt).abs() < f64::EPSILON);
    }

    #[test]
    fn test_run_respects_stop_flag() {
        let scheduler = Scheduler::new();
        let mut game_loop = GameLoop::new(scheduler, 64, 5);
        let stop_flag = Arc::new(AtomicBool::new(false));

        let stop_clone = Arc::clone(&stop_flag);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            stop_clone.store(true, Ordering::Relaxed);
        });

        game_loop.run(Arc::clone(&stop_flag));
        handle.join().unwrap();
        // If we reach here, the loop stopped correctly
    }

    #[test]
    fn test_tick_duration_precision_64hz() {
        // 1/64 = 2^(-6) is exactly representable in IEEE 754
        let scheduler = Scheduler::new();
        let game_loop = GameLoop::new(scheduler, 64, 5);
        // 15,625,000 nanoseconds = 15.625 ms
        assert_eq!(game_loop.tick_duration.as_nanos(), 15_625_000);
    }
}
