pub trait TerrainGenerator {
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool;
}
