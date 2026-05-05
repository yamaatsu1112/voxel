use voxel_engine::engine_types::Vec3;
use voxel_engine::Transform;

pub fn calculate_forward_vector(transform: &Transform) -> Vec3 {
    let yaw = transform.rotation.y.to_radians();
    Vec3::new(-yaw.sin(), 0.0, -yaw.cos())
}

pub fn calculate_right_vector(transform: &Transform) -> Vec3 {
    let yaw = transform.rotation.y.to_radians();
    Vec3::new(yaw.cos(), 0.0, -yaw.sin())
}

pub fn calculate_forward_vector_3d(transform: &Transform) -> Vec3 {
    let yaw = transform.rotation.y.to_radians();
    let pitch = transform.rotation.x.to_radians();

    Vec3::new(
        -yaw.sin() * pitch.cos(),
        pitch.sin(),
        -yaw.cos() * pitch.cos(),
    )
}
