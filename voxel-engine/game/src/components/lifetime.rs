use voxel_engine::Component;

/// Lifetime component for entities that should be destroyed after a certain time
#[derive(Component, Clone, Copy)]
pub struct Lifetime {
    pub remaining: f32, // Remaining lifetime in seconds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifetime_creation() {
        let lifetime = Lifetime { remaining: 2.0 };
        assert!((lifetime.remaining - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lifetime_decrement() {
        let mut lifetime = Lifetime { remaining: 2.0 };
        lifetime.remaining -= 0.5;
        assert!((lifetime.remaining - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lifetime_expired() {
        let mut lifetime = Lifetime { remaining: 0.1 };
        lifetime.remaining -= 0.2;
        assert!(lifetime.remaining <= 0.0);
    }
}
