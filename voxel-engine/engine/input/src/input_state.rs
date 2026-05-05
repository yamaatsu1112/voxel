use crate::input_event::InputEvent;

pub const KEY_COUNT: usize = 44;
pub const MOUSE_BUTTON_COUNT: usize = 5;

pub trait InputItem: Copy {
    fn as_index(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
    KeyG,
    KeyH,
    KeyI,
    KeyJ,
    KeyK,
    KeyL,
    KeyM,
    KeyN,
    KeyO,
    KeyP,
    KeyQ,
    KeyR,
    KeyS,
    KeyT,
    KeyU,
    KeyV,
    KeyW,
    KeyX,
    KeyY,
    KeyZ,
    KeySpace,
    KeyShiftLeft,
    KeyEscape,
    Key0,
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,
    KeyUnknown,
}

impl InputItem for Key {
    fn as_index(&self) -> usize {
        *self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    MB1,
    MB2,
    MB3,
    MBUnknown,
}

impl InputItem for MouseButton {
    fn as_index(&self) -> usize {
        *self as usize
    }
}

struct InputItemState<const N: usize> {
    pressed: [bool; N],
    just_pressed: [bool; N],
    just_released: [bool; N],
}

impl<const N: usize> InputItemState<N> {
    fn new() -> Self {
        Self {
            pressed: [false; N],
            just_pressed: [false; N],
            just_released: [false; N],
        }
    }

    fn update_states(&mut self, new_states: &[bool; N]) {
        for (i, x) in new_states.iter().enumerate() {
            self.just_pressed[i] = *x && !self.pressed[i];
        }
        for (i, x) in new_states.iter().enumerate() {
            self.just_released[i] = !(*x) && self.pressed[i];
        }
        self.pressed = *new_states;
    }

    fn is_pressed(&self, index: usize) -> bool {
        self.pressed[index]
    }

    fn is_just_pressed(&self, index: usize) -> bool {
        self.just_pressed[index]
    }

    fn is_just_released(&self, index: usize) -> bool {
        self.just_released[index]
    }
}

pub struct InputState {
    current_key_states: [bool; KEY_COUNT],
    current_mouse_button_states: [bool; MOUSE_BUTTON_COUNT],
    accumulated_mouse_delta: (f64, f64),
    key_state: InputItemState<KEY_COUNT>,
    mouse_button_state: InputItemState<MOUSE_BUTTON_COUNT>,
    mouse_position: (f64, f64),
    mouse_delta: (f64, f64),
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

impl InputState {
    pub fn new() -> Self {
        Self {
            current_key_states: [false; KEY_COUNT],
            current_mouse_button_states: [false; MOUSE_BUTTON_COUNT],
            accumulated_mouse_delta: (0.0, 0.0),
            key_state: InputItemState::new(),
            mouse_button_state: InputItemState::new(),
            mouse_position: (0.0, 0.0),
            mouse_delta: (0.0, 0.0),
        }
    }

    pub fn apply_event(&mut self, event: &InputEvent) {
        match event {
            InputEvent::KeyState { key, pressed } => {
                self.current_key_states[key.as_index()] = *pressed;
            }
            InputEvent::MousePosition(x, y) => {
                self.mouse_position = (*x, *y);
            }
            InputEvent::MouseButton { button, pressed } => {
                self.current_mouse_button_states[button.as_index()] = *pressed;
            }
            InputEvent::MouseDelta(dx, dy) => {
                self.accumulated_mouse_delta.0 += dx;
                self.accumulated_mouse_delta.1 += dy;
            }
        }
    }

    pub fn finalize_frame(&mut self) {
        self.key_state.update_states(&self.current_key_states);
        self.mouse_button_state
            .update_states(&self.current_mouse_button_states);
        self.mouse_delta = self.accumulated_mouse_delta;
        self.accumulated_mouse_delta = (0.0, 0.0);
    }

    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.key_state.is_pressed(key.as_index())
    }

    pub fn is_key_just_pressed(&self, key: Key) -> bool {
        self.key_state.is_just_pressed(key.as_index())
    }

    pub fn is_key_just_released(&self, key: Key) -> bool {
        self.key_state.is_just_released(key.as_index())
    }

    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.mouse_button_state.is_pressed(button.as_index())
    }

    pub fn is_mouse_button_just_pressed(&self, button: MouseButton) -> bool {
        self.mouse_button_state.is_just_pressed(button.as_index())
    }

    pub fn is_mouse_button_just_released(&self, button: MouseButton) -> bool {
        self.mouse_button_state.is_just_released(button.as_index())
    }

    pub fn get_mouse_position(&self) -> (f64, f64) {
        self.mouse_position
    }

    pub fn get_mouse_delta(&self) -> (f64, f64) {
        self.mouse_delta
    }
}
