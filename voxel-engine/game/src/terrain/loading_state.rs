use crate::terrain::NoiseTerrainGenerator;

pub struct TerrainLoadingState {
    generator: NoiseTerrainGenerator,
    ring_radius: i32,
    ring_offset: usize,
    max_radius: i32,
    spawn_height: i32,
}

impl TerrainLoadingState {
    pub fn new(generator: NoiseTerrainGenerator, spawn_height: i32, max_radius: i32) -> Self {
        Self {
            generator,
            ring_radius: 0,
            ring_offset: 0,
            max_radius,
            spawn_height,
        }
    }

    pub fn generator(&self) -> &NoiseTerrainGenerator {
        &self.generator
    }

    pub fn is_finished(&self) -> bool {
        self.ring_radius > self.max_radius
    }

    pub fn current_ring(&self) -> i32 {
        self.ring_radius
    }

    pub fn ring_offset(&self) -> usize {
        self.ring_offset
    }

    pub fn advance_within_ring(&mut self, processed_columns: usize, ring_len: usize) {
        self.ring_offset += processed_columns;
        if self.ring_offset >= ring_len {
            self.ring_radius += 1;
            self.ring_offset = 0;
        }
    }

    pub fn spawn_height(&self) -> i32 {
        self.spawn_height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_within_ring_rolls_to_next_ring() {
        let generator = NoiseTerrainGenerator::new();
        let mut state = TerrainLoadingState::new(generator, 0, 2);

        state.advance_within_ring(1, 1);

        assert_eq!(state.current_ring(), 1);
        assert_eq!(state.ring_offset(), 0);
    }

    #[test]
    fn advance_within_ring_tracks_partial_progress() {
        let generator = NoiseTerrainGenerator::new();
        let mut state = TerrainLoadingState::new(generator, 0, 2);

        state.advance_within_ring(3, 8);

        assert_eq!(state.current_ring(), 0);
        assert_eq!(state.ring_offset(), 3);
        assert!(!state.is_finished());
    }
}
