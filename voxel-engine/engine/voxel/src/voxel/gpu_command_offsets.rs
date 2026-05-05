use super::NUM_SVO_DATA;

/// Provides GPU command offsets per SVO chunk.
pub trait GpuCommandOffsets: Send {
    fn place_counts(&self) -> &[u32; NUM_SVO_DATA];
    fn destroy_counts(&self) -> &[u32; NUM_SVO_DATA];
    fn place_counts_mut(&mut self) -> &mut [u32; NUM_SVO_DATA];
    fn destroy_counts_mut(&mut self) -> &mut [u32; NUM_SVO_DATA];

    fn reset(&mut self) {
        self.place_counts_mut().fill(0);
        self.destroy_counts_mut().fill(0);
    }
}

pub type CommandOffsetResource = Box<dyn GpuCommandOffsets>;
