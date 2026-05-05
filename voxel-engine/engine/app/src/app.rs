use crate::ecs::scene::Scenes;
use crate::ecs::scheduler::Scheduler;
use crate::ecs::system::System;
use crate::game_loop::{DEFAULT_GAME_TICK_RATE_HZ, GameLoop};
use crate::input_event_queue::InputEventQueue;
use crate::rendering::double_buffer::DoubleBuffer;
use crate::rendering::render_graph::RenderGraph;
use crate::rendering::render_world_renderer::RenderWorldRenderer;
use crate::rendering::renderer::Renderer;
use crate::rendering::renderer_command::FrameCommandQueue;
use crate::rendering::typed_channel::{GameChannel, RenderChannel};
use crate::vulkan::record_resource::RecordResource;
use crate::vulkan::vulkan_renderer::VulkanRenderer;
use crate::window::Window;
use crate::window::WindowSize;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use engine_input::input_event::InputEvent;
use engine_input::{Key, MouseButton};
use winit::application::ApplicationHandler;
use winit::event::DeviceEvent;
use winit::event::DeviceId;
use winit::event::MouseButton as WinitMouseButton;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey;
use winit::window::{Window as WinitWindow, WindowId};

// Constants
const WINDOW_TITLE: &str = "Vulkan Test";
const _WINDOW_WIDTH: u32 = 800;
const _WINDOW_HEIGHT: u32 = 600;

fn map_key_code(key: KeyCode) -> Key {
    match key {
        KeyCode::KeyA => Key::KeyA,
        KeyCode::KeyB => Key::KeyB,
        KeyCode::KeyC => Key::KeyC,
        KeyCode::KeyD => Key::KeyD,
        KeyCode::KeyE => Key::KeyE,
        KeyCode::KeyF => Key::KeyF,
        KeyCode::KeyG => Key::KeyG,
        KeyCode::KeyH => Key::KeyH,
        KeyCode::KeyI => Key::KeyI,
        KeyCode::KeyJ => Key::KeyJ,
        KeyCode::KeyK => Key::KeyK,
        KeyCode::KeyL => Key::KeyL,
        KeyCode::KeyM => Key::KeyM,
        KeyCode::KeyN => Key::KeyN,
        KeyCode::KeyO => Key::KeyO,
        KeyCode::KeyP => Key::KeyP,
        KeyCode::KeyQ => Key::KeyQ,
        KeyCode::KeyR => Key::KeyR,
        KeyCode::KeyS => Key::KeyS,
        KeyCode::KeyT => Key::KeyT,
        KeyCode::KeyU => Key::KeyU,
        KeyCode::KeyV => Key::KeyV,
        KeyCode::KeyW => Key::KeyW,
        KeyCode::KeyX => Key::KeyX,
        KeyCode::KeyY => Key::KeyY,
        KeyCode::KeyZ => Key::KeyZ,
        KeyCode::Space => Key::KeySpace,
        KeyCode::ShiftLeft => Key::KeyShiftLeft,
        KeyCode::Escape => Key::KeyEscape,
        KeyCode::Digit0 => Key::Key0,
        KeyCode::Digit1 => Key::Key1,
        KeyCode::Digit2 => Key::Key2,
        KeyCode::Digit3 => Key::Key3,
        KeyCode::Digit4 => Key::Key4,
        KeyCode::Digit5 => Key::Key5,
        KeyCode::Digit6 => Key::Key6,
        KeyCode::Digit7 => Key::Key7,
        KeyCode::Digit8 => Key::Key8,
        KeyCode::Digit9 => Key::Key9,
        _ => Key::KeyUnknown,
    }
}

fn map_mouse_button(button: WinitMouseButton) -> MouseButton {
    match button {
        WinitMouseButton::Left => MouseButton::MB1,
        WinitMouseButton::Right => MouseButton::MB2,
        WinitMouseButton::Middle => MouseButton::MB3,
        _ => MouseButton::MBUnknown,
    }
}

pub struct App {
    window: Option<Arc<WinitWindow>>,
    scheduler: Option<Scheduler>,
    render_scheduler: Scheduler,
    vulkan_renderer: Option<VulkanRenderer>,
    stop_flag: Arc<AtomicBool>,
    game_thread: Option<JoinHandle<()>>,
    input_event_queue: Option<InputEventQueue>,
    render_input_events: Vec<InputEvent>,
    window_size_buffer: Option<Arc<DoubleBuffer<Option<WindowSize>>>>,
}

impl ApplicationHandler<()> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = WinitWindow::default_attributes().with_title(WINDOW_TITLE);
        let window = event_loop.create_window(window_attributes).unwrap();

        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);

        let winit_window = Arc::new(window);
        self.window = Some(Arc::clone(&winit_window));
        let (mut vulkan_renderer, executor) = VulkanRenderer::new(&winit_window);

        let frame_command_buffer = Arc::new(DoubleBuffer::new(
            FrameCommandQueue::default(),
            FrameCommandQueue::default(),
        ));
        vulkan_renderer.set_frame_command_buffer(Arc::clone(&frame_command_buffer));

        let renderer = Renderer::new(Arc::clone(&frame_command_buffer));

        let scheduler = self.scheduler.as_mut().expect("scheduler not set");
        let render_channel = RenderChannel::new();
        let game_channel = GameChannel::new();

        let input_event_queue = InputEventQueue::new();
        scheduler
            .get_world_mut()
            .insert_resource(input_event_queue.clone());
        self.input_event_queue = Some(input_event_queue);

        let window_inner_size = self.window.as_ref().unwrap().inner_size();
        let window_size_buffer = Arc::new(DoubleBuffer::new(None, None));
        window_size_buffer.write_and_swap(Some(WindowSize {
            width: window_inner_size.width,
            height: window_inner_size.height,
        }));
        scheduler
            .get_world_mut()
            .insert_resource(Arc::clone(&window_size_buffer));
        self.window_size_buffer = Some(Arc::clone(&window_size_buffer));

        scheduler.get_world_mut().insert_resource(renderer);
        scheduler.get_world_mut().insert_resource(executor);

        scheduler
            .get_world_mut()
            .insert_resource(render_channel.clone());
        scheduler
            .get_world_mut()
            .insert_resource(game_channel.clone());
        scheduler.update_once();

        self.render_scheduler
            .get_world_mut()
            .insert_resource(RenderWorldRenderer::new());
        self.render_scheduler
            .get_world_mut()
            .insert_resource(RenderGraph::new());
        self.render_scheduler
            .get_world_mut()
            .insert_resource(RecordResource::new());
        self.render_scheduler
            .get_world_mut()
            .insert_resource(Arc::clone(&window_size_buffer));
        self.render_scheduler
            .get_world_mut()
            .insert_resource(Window::new(
                Arc::clone(self.window.as_ref().expect("window not set")),
                Arc::clone(&self.stop_flag),
            ));
        self.render_scheduler
            .get_world_mut()
            .insert_resource(render_channel);
        self.render_scheduler
            .get_world_mut()
            .insert_resource(game_channel);
        self.render_scheduler.update_once();

        {
            let mut renderer = scheduler
                .get_world_mut()
                .get_resource_mut::<Renderer>()
                .expect("renderer not in world");
            renderer.submit_frame();
        }

        // Execute queued one-shot renderer initialization before the game thread starts.
        vulkan_renderer.flush_pending_commands();
        self.vulkan_renderer = Some(vulkan_renderer);

        // Spawn game thread
        let scheduler = self.scheduler.take().expect("scheduler not set");
        let stop_flag = Arc::clone(&self.stop_flag);
        let handle = std::thread::spawn(move || {
            let mut game_loop = GameLoop::new(scheduler, DEFAULT_GAME_TICK_RATE_HZ, 5);
            game_loop.run(stop_flag);
        });
        self.game_thread = Some(handle);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.stop_flag.store(true, Ordering::Relaxed);
                // Don't exit yet — let the event loop keep processing
                // so the game thread's pending requests can be completed.
                // RedrawRequested will detect game thread exit and call exit().
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.update_render_scheduler();
                self.flush_render_world_commands();

                if self.stop_flag.load(Ordering::Relaxed) {
                    if let Some(ref mut vr) = self.vulkan_renderer {
                        vr.process_frame();
                    }
                    if self.game_thread.as_ref().is_none_or(|h| h.is_finished()) {
                        if let Some(handle) = self.game_thread.take() {
                            let _ = handle.join();
                        }
                        event_loop.exit();
                        return;
                    }
                    // Game thread still running — keep pumping events
                    self.window.as_ref().unwrap().request_redraw();
                    return;
                }
                if let Some(ref mut vr) = self.vulkan_renderer {
                    vr.process_frame();
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                if let Some(ref mut vr) = self.vulkan_renderer {
                    vr.core.set_framebuffer_resized(true);
                }
                if let Some(ref buf) = self.window_size_buffer {
                    buf.write_and_swap(Some(WindowSize {
                        width: size.width,
                        height: size.height,
                    }));
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    let input_event = InputEvent::KeyState {
                        key: map_key_code(key_code),
                        pressed: event.state == winit::event::ElementState::Pressed,
                    };
                    self.render_input_events.push(input_event.clone());
                    if let Some(ref queue) = self.input_event_queue {
                        queue.push(input_event);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let input_event = InputEvent::MousePosition(position.x, position.y);
                self.render_input_events.push(input_event.clone());
                if let Some(ref queue) = self.input_event_queue {
                    queue.push(input_event);
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let input_event = InputEvent::MouseButton {
                    button: map_mouse_button(button),
                    pressed: state == winit::event::ElementState::Pressed,
                };
                self.render_input_events.push(input_event.clone());
                if let Some(ref queue) = self.input_event_queue {
                    queue.push(input_event);
                }
            }
            WindowEvent::MouseWheel { .. } => {}
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (x, y) } = event {
            let input_event = InputEvent::MouseDelta(x, y);
            self.render_input_events.push(input_event.clone());
            if let Some(ref queue) = self.input_event_queue {
                queue.push(input_event);
            }
        }
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            vulkan_renderer: None,
            scheduler: Some(Scheduler::new()),
            render_scheduler: Scheduler::new(),
            stop_flag: Arc::new(AtomicBool::new(false)),
            game_thread: None,
            input_event_queue: None,
            render_input_events: Vec::new(),
            window_size_buffer: None,
        }
    }

    pub fn set_scenes<T: Scenes + Clone + Send + 'static>(&mut self, scenes: T) {
        self.scheduler
            .as_mut()
            .expect("scheduler not set")
            .set_scenes(scenes);
    }

    pub fn add_first_system(&mut self, system: System) {
        self.scheduler
            .as_mut()
            .expect("scheduler not set")
            .add_first_system(system);
    }

    pub fn add_last_system(&mut self, system: System) {
        self.scheduler
            .as_mut()
            .expect("scheduler not set")
            .add_last_system(system);
    }

    pub fn add_system_once(&mut self, system: System) {
        self.scheduler
            .as_mut()
            .expect("scheduler not set")
            .add_system_once(system);
    }

    pub fn set_render_scenes<T: Scenes + Clone + Send + 'static>(&mut self, scenes: T) {
        self.render_scheduler.set_scenes(scenes);
    }

    pub fn add_render_last_system(&mut self, system: System) {
        self.render_scheduler.add_last_system(system);
    }

    pub fn add_render_first_system(&mut self, system: System) {
        self.render_scheduler.add_first_system(system);
    }

    pub fn add_render_system_once(&mut self, system: System) {
        self.render_scheduler.add_system_once(system);
    }

    pub fn join_game_thread(&mut self) {
        if let Some(handle) = self.game_thread.take() {
            let _ = handle.join();
        }
    }

    fn flush_render_world_commands(&mut self) {
        let Some(vulkan_renderer) = self.vulkan_renderer.as_mut() else {
            return;
        };

        let world = self.render_scheduler.get_world_mut();
        let frame_submit_groups = {
            let Some(mut render_graph) = world.get_resource_mut::<RenderGraph>() else {
                return;
            };
            render_graph.take()
        };
        vulkan_renderer.set_frame_submit_groups(frame_submit_groups);

        let record_resource = world
            .get_resource::<RecordResource>()
            .map(|resource| resource.into_inner().clone())
            .unwrap_or_default();
        vulkan_renderer.set_graphics_record_resource(record_resource);

        let Some(mut render_world_renderer) = world.get_resource_mut::<RenderWorldRenderer>()
        else {
            return;
        };
        vulkan_renderer
            .enqueue_frame_command_queue(render_world_renderer.frame_command_queue_mut());
    }

    fn update_render_scheduler(&mut self) {
        let events = std::mem::take(&mut self.render_input_events);
        self.render_scheduler
            .get_world_mut()
            .insert_resource(events);
        self.render_scheduler.update();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CapturedInputEvents {
        events: Vec<InputEvent>,
    }

    fn capture_render_input_events(world: &mut crate::ecs::world::World) {
        let events = {
            let Some(events) = world.get_resource::<Vec<InputEvent>>() else {
                return;
            };
            events.into_inner().clone()
        };
        world.insert_resource(CapturedInputEvents { events });
    }

    #[test]
    fn update_render_scheduler_inserts_render_input_events_before_system_update() {
        let mut app = App::new();
        app.add_render_first_system(capture_render_input_events);
        app.render_input_events
            .push(InputEvent::MouseDelta(2.0, -1.0));

        app.update_render_scheduler();

        let captured = app
            .render_scheduler
            .get_world_mut()
            .get_resource::<CapturedInputEvents>()
            .unwrap();
        assert_eq!(captured.events.len(), 1);
        assert!(matches!(
            captured.events[0],
            InputEvent::MouseDelta(2.0, -1.0)
        ));
        assert!(app.render_input_events.is_empty());
    }
}
