use voxel_engine::*;

use crate::atlas;
use crate::components::Crosshair;

const CROSSHAIR_SIZE: u32 = 32;

pub fn crosshair_setup_system(world: &mut World) {
    let ui_entity = world.create_entity();
    let window_size = world.get_resource::<WindowSize>().unwrap();
    world.add_bundle(
        ui_entity,
        (
            Crosshair,
            UI {
                position_x: window_size.width.saturating_sub(CROSSHAIR_SIZE / 2) / 2,
                position_y: window_size.height.saturating_sub(CROSSHAIR_SIZE / 2) / 2,
                width: CROSSHAIR_SIZE,
                height: CROSSHAIR_SIZE,
                atlas_offset: engine_types::Vec2::new(
                    atlas::CROSSHAIR.u_offset,
                    atlas::CROSSHAIR.v_offset,
                ),
                atlas_size: engine_types::Vec2::new(
                    atlas::CROSSHAIR.u_size,
                    atlas::CROSSHAIR.v_size,
                ),
            },
        ),
    );
}

pub fn crosshair_system(world: &mut World) {
    let mut query = world.query::<(&Crosshair, &mut UI)>();
    let window_size = world.get_resource::<WindowSize>().unwrap();

    for (_, (_, ui)) in query.iter_mut() {
        ui.position_x = window_size.width.saturating_sub(CROSSHAIR_SIZE / 2) / 2;
        ui.position_y = window_size.height.saturating_sub(CROSSHAIR_SIZE / 2) / 2;
        ui.width = CROSSHAIR_SIZE;
        ui.height = CROSSHAIR_SIZE;
    }
}

pub fn crosshair_cleanup_system(world: &mut World) {
    let query = world.query::<&Crosshair>();
    let entities_to_remove: Vec<_> = query.iter().map(|(entity, _)| entity).collect();
    for entity in entities_to_remove {
        world.delete_entity(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_engine::engine_types::Vec2;

    #[test]
    fn test_crosshair_system_centers_crosshair() {
        let mut world = World::new();

        // Set up window size resource
        world.insert_resource(WindowSize {
            width: 800,
            height: 600,
        });

        // Create crosshair entity
        let crosshair_entity = world.create_entity();
        world.add_bundle(
            crosshair_entity,
            (
                Crosshair,
                UI {
                    position_x: 0,
                    position_y: 0,
                    width: 0,
                    height: 0,
                    atlas_offset: Vec2::new(0.0, 0.0),
                    atlas_size: Vec2::new(1.0, 1.0),
                },
            ),
        );

        crosshair_system(&mut world);

        let ui = world.get_component::<UI>(crosshair_entity).unwrap();
        // Expected position: (800-16)/2 = 392, (600-16)/2 = 292
        assert_eq!(ui.position_x, 392);
        assert_eq!(ui.position_y, 292);
        assert_eq!(ui.width, 32);
        assert_eq!(ui.height, 32);
    }

    #[test]
    fn test_crosshair_system_with_different_window_size() {
        let mut world = World::new();

        world.insert_resource(WindowSize {
            width: 1920,
            height: 1080,
        });

        let crosshair_entity = world.create_entity();
        world.add_bundle(
            crosshair_entity,
            (
                Crosshair,
                UI {
                    position_x: 0,
                    position_y: 0,
                    width: 0,
                    height: 0,
                    atlas_offset: Vec2::new(0.0, 0.0),
                    atlas_size: Vec2::new(1.0, 1.0),
                },
            ),
        );

        crosshair_system(&mut world);

        let ui = world.get_component::<UI>(crosshair_entity).unwrap();
        // Expected position: (1920-16)/2 = 952, (1080-16)/2 = 532
        assert_eq!(ui.position_x, 952);
        assert_eq!(ui.position_y, 532);
        assert_eq!(ui.width, 32);
        assert_eq!(ui.height, 32);
    }
}
