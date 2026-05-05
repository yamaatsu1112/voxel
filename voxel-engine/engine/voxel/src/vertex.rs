use engine_app::engine_types::Vec2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: Vec2,
    pub tex_coord: Vec2,
}

pub const FULLSCREEN_VERTICES: &[Vertex] = &[
    Vertex {
        position: Vec2::new(-1.0, -1.0),
        tex_coord: Vec2::new(0.0, 0.0),
    },
    Vertex {
        position: Vec2::new(-1.0, 1.0),
        tex_coord: Vec2::new(0.0, 1.0),
    },
    Vertex {
        position: Vec2::new(1.0, 1.0),
        tex_coord: Vec2::new(1.0, 1.0),
    },
    Vertex {
        position: Vec2::new(1.0, -1.0),
        tex_coord: Vec2::new(1.0, 0.0),
    },
];

pub const INDICES: &[u32] = &[0, 1, 2, 2, 3, 0];
