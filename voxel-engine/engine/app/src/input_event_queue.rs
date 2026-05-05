use std::sync::Arc;

use engine_input::input_event::InputEvent;

use crate::rendering::double_buffer::DoubleBuffer;

#[derive(Clone)]
pub struct InputEventQueue {
    buffer: Arc<DoubleBuffer<Vec<InputEvent>>>,
}

impl Default for InputEventQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl InputEventQueue {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(DoubleBuffer::new(Vec::new(), Vec::new())),
        }
    }

    pub fn push(&self, event: InputEvent) {
        let mut write = self.buffer.write_buffer();
        write.push(event);
    }

    pub fn drain(&self) -> Vec<InputEvent> {
        let mut read = self.buffer.swap_and_read();
        std::mem::take(&mut *read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_drain_returns_events() {
        let queue = InputEventQueue::new();
        queue.push(InputEvent::KeyState {
            key: engine_input::Key::KeyA,
            pressed: true,
        });
        queue.push(InputEvent::MousePosition(100.0, 200.0));

        let events = queue.drain();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn drain_returns_empty_when_no_events() {
        let queue = InputEventQueue::new();
        let events = queue.drain();
        assert!(events.is_empty());
    }

    #[test]
    fn drain_clears_buffer_after_read() {
        let queue = InputEventQueue::new();
        queue.push(InputEvent::KeyState {
            key: engine_input::Key::KeyA,
            pressed: true,
        });

        let events = queue.drain();
        assert_eq!(events.len(), 1);

        let events_after = queue.drain();
        assert!(events_after.is_empty());
    }

    #[test]
    fn multiple_events_of_different_types() {
        let queue = InputEventQueue::new();
        queue.push(InputEvent::KeyState {
            key: engine_input::Key::KeyW,
            pressed: true,
        });
        queue.push(InputEvent::MousePosition(50.0, 75.0));
        queue.push(InputEvent::MouseButton {
            button: engine_input::MouseButton::MB1,
            pressed: true,
        });
        queue.push(InputEvent::MouseDelta(1.5, -2.3));

        let events = queue.drain();
        assert_eq!(events.len(), 4);
    }

    #[test]
    fn clone_shares_same_buffer() {
        let queue1 = InputEventQueue::new();
        let queue2 = queue1.clone();

        queue1.push(InputEvent::KeyState {
            key: engine_input::Key::KeyA,
            pressed: true,
        });

        let events = queue2.drain();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn concurrent_push_and_drain() {
        use std::thread;

        let queue = InputEventQueue::new();
        let writer_queue = queue.clone();
        let reader_queue = queue.clone();

        let writer = thread::spawn(move || {
            for i in 0..100 {
                writer_queue.push(InputEvent::MouseDelta(i as f64, 0.0));
            }
        });

        let reader = thread::spawn(move || {
            let mut total = 0;
            for _ in 0..200 {
                let events = reader_queue.drain();
                total += events.len();
                if total >= 100 {
                    break;
                }
                std::thread::yield_now();
            }
            total
        });

        writer.join().expect("writer thread panicked");
        let total = reader.join().expect("reader thread panicked");

        let remaining = queue.drain();
        assert_eq!(total + remaining.len(), 100);
    }
}
