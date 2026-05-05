use std::time::{Duration, Instant};

use crate::ecs::component::Component;

pub trait Interpolatable: Component {
    fn interpolate(&self, other: &Self, t: f32) -> Self;
}

#[derive(Clone)]
pub struct Interpolated<T: Interpolatable>(pub T);

#[allow(dead_code)]
pub struct InterpolationTiming {
    last_received_time: Instant,
    tick_interval: Duration,
}

#[allow(dead_code)]
impl InterpolationTiming {
    pub fn new(tick_hz: u32) -> Self {
        assert!(tick_hz > 0, "tick_hz must be greater than 0");
        Self {
            last_received_time: Instant::now(),
            tick_interval: Duration::from_secs_f64(1.0 / tick_hz as f64),
        }
    }

    pub fn mark_received(&mut self, now: Instant) {
        self.last_received_time = now;
    }

    pub fn alpha(&self, now: Instant) -> f32 {
        (now.duration_since(self.last_received_time).as_secs_f32()
            / self.tick_interval.as_secs_f32())
        .clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::InterpolationTiming;
    use std::panic;
    use std::time::{Duration, Instant};

    #[test]
    fn alpha_is_clamped_between_zero_and_one() {
        let mut timing = InterpolationTiming::new(32);
        let now = Instant::now();
        timing.mark_received(now - Duration::from_secs_f32(10.0));
        let alpha = timing.alpha(now);
        assert_eq!(alpha, 1.0);
    }

    #[test]
    fn alpha_near_zero_immediately_after_receive() {
        let mut timing = InterpolationTiming::new(32);
        let now = Instant::now();
        timing.mark_received(now);
        let alpha = timing.alpha(now);
        assert!((0.0..=0.01).contains(&alpha));
    }

    #[test]
    fn new_panics_when_tick_hz_is_zero() {
        let result = panic::catch_unwind(|| InterpolationTiming::new(0));
        assert!(result.is_err());
    }
}
