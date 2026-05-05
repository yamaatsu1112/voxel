use crate::input_state::Key;
use crate::input_state::MouseButton;

#[derive(Debug, Clone)]
pub enum InputEvent {
    KeyState { key: Key, pressed: bool },
    MousePosition(f64, f64),
    MouseButton { button: MouseButton, pressed: bool },
    MouseDelta(f64, f64),
}
