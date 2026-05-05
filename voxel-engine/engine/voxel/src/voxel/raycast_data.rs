/// Voxel raycast data resource
/// Stores the result of raycasting from the camera to the voxel world
/// Coordinates are in world space (centered at 0), already shifted from SVO space
#[derive(Debug, Clone, Copy)]
pub struct VoxelRaycastData {
    /// Voxel coordinates at the hit point (shifted to world space, centered at 0)
    pub voxel_coord: (i32, i32, i32),
    /// Normalized distance to the hit point (tc_min)
    pub distance: f32,
    /// Which face was hit (0-5: X+/X-/Y+/Y-/Z+/Z-, 6: no hit)
    pub face: u8,
}

impl Default for VoxelRaycastData {
    fn default() -> Self {
        Self {
            voxel_coord: (i32::MAX, i32::MAX, i32::MAX),
            distance: -1.0,
            face: 6, // NO_HIT
        }
    }
}

impl VoxelRaycastData {
    /// Check if the raycast hit a voxel
    pub fn is_hit(&self) -> bool {
        self.face != 6
    }
}

/// Internal raw raycast data from GPU buffer (unshifted SVO coordinates)
/// Used internally by VulkanRenderer before shifting to world space
#[derive(Debug, Clone, Copy)]
pub struct RawVoxelRaycastData {
    /// Voxel coordinates in SVO space
    pub voxel_coord: (u32, u32, u32),
    /// Normalized distance to the hit point (tc_min)
    pub distance: f32,
    /// Which face was hit (0-5: X+/X-/Y+/Y-/Z+/Z-, 6: no hit)
    pub face: u8,
}

impl RawVoxelRaycastData {
    /// Create from raw buffer data (matches shader output format)
    pub fn from_buffer(data: &[u32; 5]) -> Self {
        Self {
            voxel_coord: (data[0], data[1], data[2]),
            face: data[3] as u8,
            distance: f32::from_bits(data[4]),
        }
    }

    /// Check if the raycast hit a voxel
    pub fn is_hit(&self) -> bool {
        self.face != 6
    }
}
