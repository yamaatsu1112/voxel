/// Marker trait for resource lifetime classification.
pub trait ResourceLifetime: 'static {}

/// Persistent resource that lives across frames.
pub struct Persistent;
impl ResourceLifetime for Persistent {}

/// Per-frame resource that has separate copies for each frame in flight.
pub struct PerFrame;
impl ResourceLifetime for PerFrame {}
