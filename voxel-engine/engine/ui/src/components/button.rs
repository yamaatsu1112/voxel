use crate::ecs::component::Component;
use crate::ecs::system::System;
use engine_macro::Component;

#[derive(Component, Clone, Copy, Debug)]
pub struct Button {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub is_hovered: bool,
    pub is_pressed: bool,
    pub on_click: Option<System>,
}

impl Button {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            is_hovered: false,
            is_pressed: false,
            on_click: None,
        }
    }

    pub fn with_on_click(mut self, on_click: System) -> Self {
        self.on_click = Some(on_click);
        self
    }

    pub fn is_point_inside(&self, px: f64, py: f64) -> bool {
        let px = px as u32;
        let py = py as u32;
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}
