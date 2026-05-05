use ash::vk;
use std::marker::PhantomData;
use std::ptr;

use super::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT;

const MAX_TIMESTAMPS_PER_FRAME: u32 = 64;
const LOG_INTERVAL_SECS: f64 = 5.0;

pub struct GpuTimingManager {
    query_pools: [vk::QueryPool; MAX_FRAMES_IN_FLIGHT],
    query_count: [u32; MAX_FRAMES_IN_FLIGHT],
    labels: [Vec<(String, u32, u32)>; MAX_FRAMES_IN_FLIGHT],
    timestamp_period: f32,
    aggregator: TimingAggregator,
}

impl GpuTimingManager {
    pub fn max_timestamps_per_frame() -> u32 {
        MAX_TIMESTAMPS_PER_FRAME
    }

    pub fn new(device: &ash::Device, timestamp_period: f32) -> Self {
        let query_pools = std::array::from_fn(|_| {
            let create_info = vk::QueryPoolCreateInfo {
                s_type: vk::StructureType::QUERY_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::QueryPoolCreateFlags::empty(),
                query_type: vk::QueryType::TIMESTAMP,
                query_count: MAX_TIMESTAMPS_PER_FRAME,
                pipeline_statistics: vk::QueryPipelineStatisticFlags::empty(),
                _marker: PhantomData,
            };
            unsafe {
                device
                    .create_query_pool(&create_info, None)
                    .expect("Failed to create timestamp query pool")
            }
        });

        Self {
            query_pools,
            query_count: [0; MAX_FRAMES_IN_FLIGHT],
            labels: std::array::from_fn(|_| Vec::new()),
            timestamp_period,
            aggregator: TimingAggregator::new(),
        }
    }

    pub fn query_pool(&self, frame: usize) -> vk::QueryPool {
        self.query_pools[frame]
    }

    pub fn query_count(&self, frame: usize) -> u32 {
        self.query_count[frame]
    }

    fn reserve_begin_query(&mut self, frame: usize, label: &str) -> Option<u32> {
        let count = self.query_count[frame];
        if count + 1 >= MAX_TIMESTAMPS_PER_FRAME {
            return None;
        }

        let begin_index = count;
        self.query_count[frame] = count + 1;
        self.labels[frame].push((label.to_string(), begin_index, u32::MAX));
        Some(begin_index)
    }

    fn reserve_end_query(&mut self, frame: usize, label: &str) -> Option<u32> {
        let count = self.query_count[frame];
        if count >= MAX_TIMESTAMPS_PER_FRAME {
            return None;
        }

        let entry = self.labels[frame]
            .iter_mut()
            .rev()
            .find(|(existing_label, _, end)| existing_label == label && *end == u32::MAX)?;
        let end_index = count;
        self.query_count[frame] = count + 1;
        entry.2 = end_index;
        Some(end_index)
    }

    pub fn cmd_begin_timing(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame: usize,
        label: &str,
    ) {
        let Some(begin_index) = self.reserve_begin_query(frame, label) else {
            return;
        };

        unsafe {
            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pools[frame],
                begin_index,
            );
        }
    }

    pub fn cmd_end_timing(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame: usize,
        label: &str,
    ) {
        let Some(end_index) = self.reserve_end_query(frame, label) else {
            return;
        };

        unsafe {
            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pools[frame],
                end_index,
            );
        }
    }

    pub fn collect_results(&mut self, device: &ash::Device, frame: usize) {
        let count = self.query_count[frame];
        if count == 0 {
            return;
        }

        let mut timestamps = vec![0u64; count as usize];
        let result = unsafe {
            device.get_query_pool_results(
                self.query_pools[frame],
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64,
            )
        };

        if result.is_ok() {
            for (label, begin_index, end_index) in &self.labels[frame] {
                if *end_index == u32::MAX {
                    continue;
                }
                let begin_ts = timestamps[*begin_index as usize];
                let end_ts = timestamps[*end_index as usize];
                let duration_ns =
                    (end_ts.wrapping_sub(begin_ts)) as f64 * self.timestamp_period as f64;
                let duration_ms = duration_ns / 1_000_000.0;
                self.aggregator.add_sample(label, duration_ms);
            }
        }

        self.query_count[frame] = 0;
        self.labels[frame].clear();
    }

    pub fn maybe_log(&mut self) {
        self.aggregator.maybe_log();
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for pool in &self.query_pools {
            unsafe {
                device.destroy_query_pool(*pool, None);
            }
        }
    }
}

struct TimingEntry {
    samples: Vec<f64>,
}

struct TimingAggregator {
    entries: std::collections::HashMap<String, TimingEntry>,
    last_log_time: std::time::Instant,
}

impl TimingAggregator {
    fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            last_log_time: std::time::Instant::now(),
        }
    }

    fn add_sample(&mut self, label: &str, duration_ms: f64) {
        self.entries
            .entry(label.to_string())
            .or_insert_with(|| TimingEntry {
                samples: Vec::new(),
            })
            .samples
            .push(duration_ms);
    }

    fn maybe_log(&mut self) {
        if self.last_log_time.elapsed().as_secs_f64() < LOG_INTERVAL_SECS {
            return;
        }
        self.last_log_time = std::time::Instant::now();

        let mut labels: Vec<_> = self.entries.keys().cloned().collect();
        labels.sort();

        for label in &labels {
            if let Some(entry) = self.entries.get(label) {
                if entry.samples.is_empty() {
                    continue;
                }
                let sum: f64 = entry.samples.iter().sum();
                let avg = sum / entry.samples.len() as f64;
                let min = entry.samples.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = entry
                    .samples
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                log::info!(
                    "[GPU Timing] {}: avg={:.2}ms, min={:.2}ms, max={:.2}ms",
                    label,
                    avg,
                    min,
                    max
                );
            }
        }

        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_aggregator_collects_samples() {
        let mut aggregator = TimingAggregator::new();
        aggregator.add_sample("test", 1.0);
        aggregator.add_sample("test", 2.0);
        aggregator.add_sample("test", 3.0);

        let entry = aggregator.entries.get("test").unwrap();
        assert_eq!(entry.samples.len(), 3);
        assert_eq!(entry.samples[0], 1.0);
        assert_eq!(entry.samples[1], 2.0);
        assert_eq!(entry.samples[2], 3.0);
    }

    #[test]
    fn timing_aggregator_clears_after_log() {
        let mut aggregator = TimingAggregator::new();
        aggregator.add_sample("test", 1.0);
        // Force the log time to be in the past
        aggregator.last_log_time = std::time::Instant::now() - std::time::Duration::from_secs(2);
        aggregator.maybe_log();
        assert!(aggregator.entries.is_empty());
    }

    #[test]
    fn timing_aggregator_skips_log_before_interval() {
        let mut aggregator = TimingAggregator::new();
        aggregator.add_sample("test", 1.0);
        aggregator.maybe_log();
        assert!(!aggregator.entries.is_empty());
    }

    #[test]
    fn max_timestamps_constant() {
        assert_eq!(MAX_TIMESTAMPS_PER_FRAME, 64);
    }

    #[test]
    fn reserve_end_query_is_noop_without_matching_begin() {
        let mut manager = GpuTimingManager {
            query_pools: [vk::QueryPool::null(); MAX_FRAMES_IN_FLIGHT],
            query_count: [0; MAX_FRAMES_IN_FLIGHT],
            labels: std::array::from_fn(|_| Vec::new()),
            timestamp_period: 1.0,
            aggregator: TimingAggregator::new(),
        };

        assert_eq!(manager.reserve_end_query(0, "missing"), None);
        assert_eq!(manager.query_count(0), 0);
    }

    #[test]
    fn reserve_end_query_is_noop_after_begin_overflow() {
        let mut manager = GpuTimingManager {
            query_pools: [vk::QueryPool::null(); MAX_FRAMES_IN_FLIGHT],
            query_count: [MAX_TIMESTAMPS_PER_FRAME - 1, 0],
            labels: std::array::from_fn(|_| Vec::new()),
            timestamp_period: 1.0,
            aggregator: TimingAggregator::new(),
        };

        assert_eq!(manager.reserve_begin_query(0, "overflow"), None);
        assert_eq!(manager.reserve_end_query(0, "overflow"), None);
        assert_eq!(manager.query_count(0), MAX_TIMESTAMPS_PER_FRAME - 1);
        assert!(manager.labels[0].is_empty());
    }
}
