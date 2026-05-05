use crate::builders::configs::buffer_configs::UIInstanceBuffer;
use crate::components::ui::UI;
use crate::components::ui::UIInstance;
use crate::ecs::query::QueryExt;
use crate::recordables::ui::UIInstanceCount;
use crate::rendering::NoMutex;
use crate::rendering::commands::UploadToPerFrameBufferCommand;
use crate::rendering::render_world_renderer::RenderWorldRenderer;
use crate::vulkan::record_resource::RecordResource;
use crate::window::WindowSize;

pub fn ui_system(world: &mut crate::ecs::world::World) {
    let query = world.query::<&UI>();
    let window_size = {
        let ws = world.get_resource::<WindowSize>().unwrap();
        (ws.width, ws.height)
    };
    let ui_components = query.iter().map(|(_, ui)| *ui).collect::<Vec<UI>>();
    let ui_instances = build_ui_instances(&ui_components, window_size);
    world
        .get_resource_mut::<RecordResource>()
        .unwrap()
        .insert(UIInstanceCount(ui_instances.len() as u32));
    if ui_instances.is_empty() {
        return;
    }
    world
        .get_resource_mut::<RenderWorldRenderer>()
        .unwrap()
        .add_command(UploadToPerFrameBufferCommand::<
            NoMutex,
            UIInstanceBuffer,
            UIInstance,
        >::new(0, &ui_instances));
}

fn build_ui_instances(ui_components: &[UI], window_size: (u32, u32)) -> Vec<UIInstance> {
    ui_components
        .iter()
        .map(|ui_component| ui_component_to_instance(ui_component, window_size))
        .collect::<Vec<UIInstance>>()
}

fn ui_component_to_instance(ui_component: &UI, window_size: (u32, u32)) -> UIInstance {
    let position_x = ui_component.position_x as f32 / (window_size.0 >> 1) as f32;
    let position_y = ui_component.position_y as f32 / (window_size.1 >> 1) as f32;
    let width = ui_component.width as f32 / window_size.0 as f32;
    let height = ui_component.height as f32 / window_size.1 as f32;
    UIInstance {
        position_x,
        position_y,
        width,
        height,
        atlas_u_offset: ui_component.atlas_offset.x,
        atlas_v_offset: ui_component.atlas_offset.y,
        atlas_u_size: ui_component.atlas_size.x,
        atlas_v_size: ui_component.atlas_size.y,
    }
}

#[cfg(test)]
mod tests {
    use super::{build_ui_instances, ui_component_to_instance};
    use crate::components::ui::UI;
    use engine_app::engine_types::Vec2;

    #[test]
    fn build_ui_instances_keeps_all_components() {
        let ui_components = (0..40).map(|_| UI::default()).collect::<Vec<UI>>();

        let instances = build_ui_instances(&ui_components, (800, 600));

        assert_eq!(instances.len(), ui_components.len());
    }

    #[test]
    fn ui_component_to_instance_converts_screen_coordinates() {
        let ui_component = UI {
            position_x: 200,
            position_y: 150,
            width: 80,
            height: 60,
            atlas_offset: Vec2::new(0.1, 0.2),
            atlas_size: Vec2::new(0.3, 0.4),
        };

        let instance = ui_component_to_instance(&ui_component, (800, 600));

        assert_eq!(instance.position_x, 0.5);
        assert_eq!(instance.position_y, 0.5);
        assert_eq!(instance.width, 0.1);
        assert_eq!(instance.height, 0.1);
        assert_eq!(instance.atlas_u_offset, 0.1);
        assert_eq!(instance.atlas_v_offset, 0.2);
        assert_eq!(instance.atlas_u_size, 0.3);
        assert_eq!(instance.atlas_v_size, 0.4);
    }
}
