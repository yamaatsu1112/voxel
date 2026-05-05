use voxel_engine::Component;

#[derive(Component, Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct TPSCameraSettings {
    pub distance: f32,
    pub height_offset: f32,
    pub pitch_min: f32,
    pub pitch_max: f32,
}

impl Default for TPSCameraSettings {
    fn default() -> Self {
        Self {
            distance: 5.0,
            height_offset: 1.0,
            pitch_min: -60.0,
            pitch_max: 60.0,
        }
    }
}
