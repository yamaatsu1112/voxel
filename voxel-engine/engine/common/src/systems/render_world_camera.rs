use crate::buffer_configs::{ComputeUniformBuffer, UniformBuffer};
use crate::components::local_transform::LocalTransform;
use crate::components::transform::Transform;
use crate::ecs::component::Component;
use crate::ecs::query::QueryExt;
use crate::ecs::world::World;
use crate::game_loop::DeltaTime;
use crate::rendering::NoMutex;
use crate::rendering::commands::{UploadToBufferCommand, UploadToPerFrameBufferCommand};
use crate::rendering::render_world_renderer::RenderWorldRenderer;
use crate::rendering::typed_channel::GameChannel;
use crate::systems::camera::CameraTransformData;
use crate::uniform_buffer_object::UniformBufferObject;
use crate::window::WindowSize;

use engine_app::WorldLinkId;
use engine_app::engine_types::{Mat4, Vec2, Vec3};
use engine_input::InputState;
use engine_macro::Component;

#[derive(Clone, Default)]
pub struct CameraRenderData {
    pub items: Vec<CameraRenderItem>,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraRenderItem {
    pub world_link_id: WorldLinkId,
    pub player_transform: Transform,
}

#[derive(Component, Clone, Copy, Default, Debug)]
pub(crate) struct CameraRenderTarget;

#[derive(Component, Clone, Copy, Default, Debug)]
pub(crate) struct RenderCamera;

const CAMERA_SENSITIVITY: f32 = 50.0;
const CAMERA_PITCH_MIN: f32 = -85.0;
const CAMERA_PITCH_MAX: f32 = 85.0;
const CAMERA_DISTANCE: f32 = 2.0;
const CAMERA_HEIGHT_OFFSET: f32 = 0.5;
const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 100.0;
const CAMERA_FOV_DEGREES: f32 = 90.0;

pub(crate) fn setup_render_camera_system(world: &mut World) {
    let camera_entity = world.create_entity();
    world.add_bundle(camera_entity, (RenderCamera, Transform::default()));
}

pub fn render_camera_system(world: &mut World) {
    let mouse_delta = world
        .get_resource::<InputState>()
        .map(|input_state| input_state.get_mouse_delta())
        .unwrap_or((0.0, 0.0));
    let dt = world
        .get_resource::<DeltaTime>()
        .map(|delta| delta.dt as f32)
        .unwrap_or(0.0);

    let player_transform = {
        let query = world.query::<(&CameraRenderTarget, &Transform)>();
        query.iter().next().map(|(_, (_, transform))| *transform)
    };
    let Some(player_transform) = player_transform else {
        return;
    };

    let camera_entity = {
        let query = world.query::<(&RenderCamera, &Transform)>();
        query.iter().next().map(|(entity, _)| entity)
    };
    let Some(camera_entity) = camera_entity else {
        return;
    };

    let (camera_transform, offset) = {
        let Some(camera_transform) = world.get_component_mut::<Transform>(camera_entity) else {
            return;
        };
        update_rotation(camera_transform, mouse_delta, dt);

        let offset = compute_tps_follow_offset(
            camera_transform.rotation.x,
            camera_transform.rotation.y,
            CAMERA_DISTANCE,
            CAMERA_HEIGHT_OFFSET,
        );
        camera_transform.position = Vec3::new(
            player_transform.position.x + offset.0,
            player_transform.position.y + offset.1,
            player_transform.position.z + offset.2,
        );

        (*camera_transform, offset)
    };

    let window_size = world
        .get_resource::<WindowSize>()
        .map(|window_size| (window_size.width, window_size.height));
    let Some(window_size) = window_size else {
        return;
    };
    if window_size.0 == 0 || window_size.1 == 0 {
        return;
    }

    let mut renderer = world.get_resource_mut::<RenderWorldRenderer>();
    let Some(renderer) = renderer.as_mut() else {
        return;
    };

    let ubo = compute_ubo(
        &camera_transform,
        window_size,
        CAMERA_NEAR,
        CAMERA_FAR,
        CAMERA_FOV_DEGREES,
    );

    renderer.add_command(
        UploadToPerFrameBufferCommand::<NoMutex, UniformBuffer, _>::new(
            0,
            std::slice::from_ref(&ubo),
        ),
    );
    renderer.add_command(
        UploadToBufferCommand::<NoMutex, ComputeUniformBuffer, _>::new(
            0,
            std::slice::from_ref(&ubo),
            0,
        ),
    );

    if let Some(game_channel) = world.get_resource::<GameChannel>() {
        let game_channel = GameChannel::clone(&game_channel);
        game_channel.write::<CameraTransformData, _>(|value| {
            *value = Some(CameraTransformData {
                rotation_x: camera_transform.rotation.x,
                rotation_y: camera_transform.rotation.y,
                local_transform: LocalTransform::new(Vec3::new(offset.0, offset.1, offset.2)),
            });
        });
    }
}

fn update_rotation(transform: &mut Transform, mouse_delta: (f64, f64), dt: f32) {
    transform.rotation.y -= CAMERA_SENSITIVITY * mouse_delta.0 as f32 * dt;
    transform.rotation.x -= CAMERA_SENSITIVITY * mouse_delta.1 as f32 * dt;
    transform.rotation.x = transform
        .rotation
        .x
        .clamp(CAMERA_PITCH_MIN, CAMERA_PITCH_MAX);
}

fn compute_tps_follow_offset(
    rotation_x: f32,
    rotation_y: f32,
    distance: f32,
    height_offset: f32,
) -> (f32, f32, f32) {
    let pitch = rotation_x.to_radians();
    let yaw = rotation_y.to_radians();
    let offset_x = yaw.sin() * pitch.cos() * distance;
    let offset_y = height_offset - pitch.sin() * distance;
    let offset_z = yaw.cos() * pitch.cos() * distance;
    (offset_x, offset_y, offset_z)
}

fn compute_ubo(
    transform: &Transform,
    window_size: (u32, u32),
    near: f32,
    far: f32,
    fov_degrees: f32,
) -> UniformBufferObject {
    let camera_direction = calculate_camera_direction(transform);
    let camera_position = transform.position;
    let camera_target = Vec3::new(
        transform.position.x + camera_direction.0,
        transform.position.y + camera_direction.1,
        transform.position.z + camera_direction.2,
    );

    let view = Mat4::look_at_rh(camera_position, camera_target, Vec3::new(0.0, 0.1, 0.0));
    let mut proj = Mat4::perspective_rh_gl(
        fov_degrees.to_radians(),
        window_size.0 as f32 / window_size.1 as f32,
        near,
        far,
    );
    proj.col_mut(1).y *= -1.0;
    let view_proj = proj * view;
    let inv_view_proj = view_proj.inverse();
    let sky_light_direction = Vec3::new(1.1, 1.2, 1.0).normalize();

    UniformBufferObject {
        view,
        proj,
        inv_view_proj,
        camera_position,
        _pad0: 0.0,
        sky_light_direction,
        _pad1: 0.0,
        window_size: Vec2::new(window_size.0 as f32, window_size.1 as f32),
    }
}

fn calculate_camera_direction(transform: &Transform) -> (f32, f32, f32) {
    let yaw = transform.rotation.y.to_radians();
    let pitch = transform.rotation.x.to_radians();
    let x = -yaw.sin() * pitch.cos();
    let y = pitch.sin();
    let z = -yaw.cos() * pitch.cos();
    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::{
        CAMERA_HEIGHT_OFFSET, CAMERA_PITCH_MAX, CAMERA_PITCH_MIN, compute_tps_follow_offset,
        compute_ubo, update_rotation,
    };
    use crate::components::transform::Transform;
    use engine_app::engine_types::{Vec2, Vec3};

    #[test]
    fn update_rotation_clamps_pitch_to_max() {
        let mut transform = Transform {
            position: Vec3::ZERO,
            rotation: Vec3::new(CAMERA_PITCH_MAX - 0.1, 0.0, 0.0),
        };

        update_rotation(&mut transform, (0.0, -10.0), 1.0);

        assert_eq!(transform.rotation.x, CAMERA_PITCH_MAX);
    }

    #[test]
    fn update_rotation_clamps_pitch_to_min() {
        let mut transform = Transform {
            position: Vec3::ZERO,
            rotation: Vec3::new(CAMERA_PITCH_MIN + 0.1, 0.0, 0.0),
        };

        update_rotation(&mut transform, (0.0, 10.0), 1.0);

        assert_eq!(transform.rotation.x, CAMERA_PITCH_MIN);
    }

    #[test]
    fn update_rotation_no_dt_keeps_values() {
        let mut transform = Transform {
            position: Vec3::ZERO,
            rotation: Vec3::new(12.0, -18.0, 0.0),
        };

        update_rotation(&mut transform, (3.0, -2.0), 0.0);

        assert_eq!(transform.rotation.x, 12.0);
        assert_eq!(transform.rotation.y, -18.0);
    }

    #[test]
    fn compute_tps_follow_offset_matches_forward_convention() {
        let offset = compute_tps_follow_offset(0.0, 0.0, 2.0, CAMERA_HEIGHT_OFFSET);
        assert!((offset.0 - 0.0).abs() < 0.001);
        assert!((offset.1 - CAMERA_HEIGHT_OFFSET).abs() < 0.001);
        assert!((offset.2 - 2.0).abs() < 0.001);
    }

    #[test]
    fn compute_tps_follow_offset_handles_pitch_extreme() {
        let offset = compute_tps_follow_offset(60.0, 0.0, 2.0, CAMERA_HEIGHT_OFFSET);
        let expected_y = CAMERA_HEIGHT_OFFSET - (60.0_f32).to_radians().sin() * 2.0;
        let expected_z = (60.0_f32).to_radians().cos() * 2.0;

        assert!((offset.1 - expected_y).abs() < 0.001);
        assert!((offset.2 - expected_z).abs() < 0.001);
    }

    #[test]
    fn compute_ubo_preserves_position_and_window_size() {
        let transform = Transform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
        };

        let ubo = compute_ubo(&transform, (1280, 720), 0.1, 100.0, 90.0);

        assert_eq!(ubo.camera_position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(ubo.window_size, Vec2::new(1280.0, 720.0));
    }
}
