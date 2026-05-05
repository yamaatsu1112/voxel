use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use winit::window::CursorGrabMode;

#[derive(Debug, Clone, Copy)]
pub struct WindowSize {
    pub width: u32,
    pub height: u32,
}

pub struct Window {
    winit_window: Arc<winit::window::Window>,
    stop_flag: Arc<AtomicBool>,
}

impl Window {
    pub fn new(winit_window: Arc<winit::window::Window>, stop_flag: Arc<AtomicBool>) -> Self {
        Self {
            winit_window,
            stop_flag,
        }
    }

    pub fn enable_cursor_grab(&self) {
        let _ = self.winit_window.set_cursor_grab(CursorGrabMode::Locked);
        self.winit_window.set_cursor_visible(false);
    }

    pub fn disable_cursor_grab(&self) {
        let _ = self.winit_window.set_cursor_grab(CursorGrabMode::None);
        self.winit_window.set_cursor_visible(true);
    }

    pub fn close(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }
}
