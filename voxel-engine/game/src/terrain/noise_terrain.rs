use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::terrain::TerrainGenerator;

const NOISE_FREQUENCY: f64 = 0.005;
const NOISE_AMPLITUDE: f64 = 128.0;
const NOISE_OCTAVES: usize = 6;

pub struct NoiseTerrainGenerator {
    noise: Fbm<Perlin>,
}

impl NoiseTerrainGenerator {
    pub fn new() -> Self {
        Self {
            noise: Fbm::<Perlin>::new(0)
                .set_octaves(NOISE_OCTAVES)
                .set_frequency(NOISE_FREQUENCY),
        }
    }

    pub fn surface_height(&self, x: i32, z: i32) -> i32 {
        (self.noise.get([x as f64, z as f64]) * NOISE_AMPLITUDE).round() as i32
    }
}

impl TerrainGenerator for NoiseTerrainGenerator {
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        y <= self.surface_height(x, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_height_is_deterministic() {
        let terrain = NoiseTerrainGenerator::new();

        assert_eq!(terrain.surface_height(0, 0), terrain.surface_height(0, 0));
        assert_eq!(
            terrain.surface_height(-512, 127),
            terrain.surface_height(-512, 127)
        );
    }

    #[test]
    fn is_solid_matches_surface_height() {
        let terrain = NoiseTerrainGenerator::new();
        let height = terrain.surface_height(0, 0);

        assert!(terrain.is_solid(0, height, 0));
        assert!(!terrain.is_solid(0, height + 1, 0));
    }
}
