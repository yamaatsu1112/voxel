use engine_app::engine_types::{Mat4, Vec2, Vec3};

#[repr(C)]
#[derive(Clone)]
pub struct UniformBufferObject {
    pub view: Mat4,
    pub proj: Mat4,
    pub inv_view_proj: Mat4,
    pub camera_position: Vec3,
    pub _pad0: f32,
    pub sky_light_direction: Vec3,
    pub _pad1: f32,
    pub window_size: Vec2,
}
