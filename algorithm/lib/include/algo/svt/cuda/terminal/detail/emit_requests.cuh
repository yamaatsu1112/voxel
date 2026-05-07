#pragma once

#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/terminal/detail/common.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

constexpr std::uint32_t kTerminalAxisX = 0u;
constexpr std::uint32_t kTerminalAxisY = 1u;
constexpr std::uint32_t kTerminalAxisZ = 2u;

inline constexpr std::uint32_t kTerminalLeafBrickGroupCapacity = 8u;

struct TerminalAxisInterval {
  std::int32_t start;
  std::uint32_t log_size;
};

__device__ inline std::uint32_t
terminal_decompose_axis_intervals_device(std::int32_t begin, std::int32_t end,
                                         TerminalAxisInterval *intervals,
                                         std::uint32_t suffix[kMaxDepth + 2u]) {
  for (std::uint32_t log_size = 0u; log_size <= kMaxDepth + 1u; ++log_size)
    suffix[log_size] = 0u;

  std::uint32_t count = 0u;
  std::uint32_t current = static_cast<std::uint32_t>(begin);
  const std::uint32_t limit = static_cast<std::uint32_t>(end);
  const std::uint32_t max_size = 1u << kMaxTerminalFullCellLogSize;
  while (current < limit) {
    const std::uint32_t remaining = limit - current;
    std::uint32_t aligned = current == 0u ? max_size : current & -current;
    if (aligned > max_size)
      aligned = max_size;
    const std::uint32_t fit = 1u << (31u - __clz(remaining));
    const std::uint32_t size = aligned < fit ? aligned : fit;
    intervals[count++] = {
        static_cast<std::int32_t>(current),
        static_cast<std::uint32_t>(31u - __clz(size)),
    };
    ++suffix[31u - __clz(size)];
    current += size;
  }
  return count;
}

__device__ inline std::uint32_t
terminal_axis_exact_length_device(
    const std::uint32_t suffix[kMaxDepth + 2u], std::uint32_t log_size) {
  return suffix[log_size] - suffix[log_size + 1u];
}

__device__ inline std::uint32_t
terminal_axis_larger_length_device(
    const std::uint32_t suffix[kMaxDepth + 2u], std::uint32_t log_size) {
  return suffix[log_size + 1u];
}

__device__ inline std::uint32_t terminal_axis_exact_slot_count_device(
    const std::uint32_t suffix[kMaxDepth + 2u], std::uint32_t log_size) {
  return terminal_axis_exact_length_device(suffix, log_size) >> log_size;
}

__device__ inline std::uint32_t terminal_axis_larger_slot_count_device(
    const std::uint32_t suffix[kMaxDepth + 2u], std::uint32_t log_size) {
  return terminal_axis_larger_length_device(suffix, log_size) >> log_size;
}

__device__ inline std::int32_t terminal_select_exact_axis_slot_device(
    const TerminalAxisInterval *intervals, std::uint32_t interval_count,
    std::uint32_t log_size, std::uint32_t index) {
  for (std::uint32_t interval_index = 0u; interval_index < interval_count;
       ++interval_index) {
    const TerminalAxisInterval interval = intervals[interval_index];
    if (interval.log_size != log_size)
      continue;
    if (index == 0u)
      return interval.start;
    --index;
  }
  return 0;
}

__device__ inline std::int32_t terminal_select_larger_axis_slot_device(
    const TerminalAxisInterval *intervals, std::uint32_t interval_count,
    std::uint32_t log_size, std::uint32_t index) {
  const std::uint32_t slot_size = 1u << log_size;
  for (std::uint32_t interval_index = 0u; interval_index < interval_count;
       ++interval_index) {
    const TerminalAxisInterval interval = intervals[interval_index];
    if (interval.log_size <= log_size)
      continue;
    const std::uint32_t slot_count = 1u << (interval.log_size - log_size);
    if (index >= slot_count) {
      index -= slot_count;
      continue;
    }
    return interval.start + static_cast<std::int32_t>(index * slot_size);
  }
  return 0;
}

__device__ inline std::int32_t terminal_select_axis_slot_device(
    const TerminalAxisInterval *intervals, std::uint32_t interval_count,
    std::uint32_t log_size, bool exact, std::uint32_t index) {
  return exact ? terminal_select_exact_axis_slot_device(
                     intervals, interval_count, log_size, index)
               : terminal_select_larger_axis_slot_device(
                     intervals, interval_count, log_size, index);
}

__device__ inline bool terminal_axis_mask_is_exact(std::uint32_t mask,
                                                   std::uint32_t axis) {
  return ((mask >> axis) & 1u) != 0u;
}

__device__ inline std::uint32_t
terminal_axis_mask_request_count_device(
    std::uint32_t mask, std::uint32_t log_size,
    const std::uint32_t suffix_x[kMaxDepth + 2u],
    const std::uint32_t suffix_y[kMaxDepth + 2u],
    const std::uint32_t suffix_z[kMaxDepth + 2u]) {
  const std::uint32_t length_x =
      terminal_axis_mask_is_exact(mask, kTerminalAxisX)
          ? terminal_axis_exact_length_device(suffix_x, log_size)
          : terminal_axis_larger_length_device(suffix_x, log_size);
  const std::uint32_t length_y =
      terminal_axis_mask_is_exact(mask, kTerminalAxisY)
          ? terminal_axis_exact_length_device(suffix_y, log_size)
          : terminal_axis_larger_length_device(suffix_y, log_size);
  const std::uint32_t length_z =
      terminal_axis_mask_is_exact(mask, kTerminalAxisZ)
          ? terminal_axis_exact_length_device(suffix_z, log_size)
          : terminal_axis_larger_length_device(suffix_z, log_size);
  return (length_x * length_y * length_z) >> (kGroupSizeExp * log_size);
}

__device__ inline std::uint32_t terminal_axis_mask_slot_count_device(
    std::uint32_t mask, std::uint32_t axis, std::uint32_t log_size,
    const std::uint32_t suffix_x[kMaxDepth + 2u],
    const std::uint32_t suffix_y[kMaxDepth + 2u],
    const std::uint32_t suffix_z[kMaxDepth + 2u]) {
  const bool exact = terminal_axis_mask_is_exact(mask, axis);
  if (axis == kTerminalAxisX)
    return exact ? terminal_axis_exact_slot_count_device(suffix_x, log_size)
                 : terminal_axis_larger_slot_count_device(suffix_x, log_size);
  if (axis == kTerminalAxisY)
    return exact ? terminal_axis_exact_slot_count_device(suffix_y, log_size)
                 : terminal_axis_larger_slot_count_device(suffix_y, log_size);
  return exact ? terminal_axis_exact_slot_count_device(suffix_z, log_size)
               : terminal_axis_larger_slot_count_device(suffix_z, log_size);
}

__device__ inline TerminalRequest
select_terminal_full_request_device(const LeafBox &box,
                                    std::uint32_t local_index) {
  constexpr std::uint32_t kMaxAxisIntervals = kMaxDepth * 2u + 2u;
  TerminalAxisInterval intervals_x[kMaxAxisIntervals]{};
  TerminalAxisInterval intervals_y[kMaxAxisIntervals]{};
  TerminalAxisInterval intervals_z[kMaxAxisIntervals]{};
  std::uint32_t suffix_x[kMaxDepth + 2u]{};
  std::uint32_t suffix_y[kMaxDepth + 2u]{};
  std::uint32_t suffix_z[kMaxDepth + 2u]{};
  const std::uint32_t interval_count_x =
      terminal_decompose_axis_intervals_device(box.x0, box.x1, intervals_x,
                                               suffix_x);
  const std::uint32_t interval_count_y =
      terminal_decompose_axis_intervals_device(box.y0, box.y1, intervals_y,
                                               suffix_y);
  const std::uint32_t interval_count_z =
      terminal_decompose_axis_intervals_device(box.z0, box.z1, intervals_z,
                                               suffix_z);
  terminal_axis_histogram_to_suffix_lengths_device(suffix_x);
  terminal_axis_histogram_to_suffix_lengths_device(suffix_y);
  terminal_axis_histogram_to_suffix_lengths_device(suffix_z);

  std::uint32_t selected_log_size = 0u;
  // The final log_size=0 bucket is the residual case, so no count check is
  // needed for it.
  for (std::int32_t candidate =
           static_cast<std::int32_t>(kMaxTerminalFullCellLogSize);
       candidate > 0; --candidate) {
    const std::uint32_t log_size = static_cast<std::uint32_t>(candidate);
    const std::uint32_t count = terminal_full_count_for_log_size_device(
        suffix_x, suffix_y, suffix_z, log_size);
    if (local_index < count) {
      selected_log_size = log_size;
      break;
    }
    local_index -= count;
  }

  constexpr std::uint32_t kMaskOrder[7] = {1u, 2u, 4u, 3u, 5u, 6u, 7u};
  std::uint32_t selected_mask = kMaskOrder[6];
  // The final mask=7 bucket is the residual case, so no count check is needed
  // for it.
  for (std::uint32_t mask_index = 0u; mask_index < 6u; ++mask_index) {
    const std::uint32_t mask = kMaskOrder[mask_index];
    const std::uint32_t count = terminal_axis_mask_request_count_device(
        mask, selected_log_size, suffix_x, suffix_y, suffix_z);
    if (local_index < count) {
      selected_mask = mask;
      break;
    }
    local_index -= count;
  }

  const std::uint32_t count_y = terminal_axis_mask_slot_count_device(
      selected_mask, kTerminalAxisY, selected_log_size, suffix_x, suffix_y,
      suffix_z);
  const std::uint32_t count_z = terminal_axis_mask_slot_count_device(
      selected_mask, kTerminalAxisZ, selected_log_size, suffix_x, suffix_y,
      suffix_z);
  const std::uint32_t slot_x = local_index / (count_y * count_z);
  const std::uint32_t rem = local_index - slot_x * count_y * count_z;
  const std::uint32_t slot_y = rem / count_z;
  const std::uint32_t slot_z = rem - slot_y * count_z;

  const std::int32_t x = terminal_select_axis_slot_device(
      intervals_x, interval_count_x, selected_log_size,
      terminal_axis_mask_is_exact(selected_mask, kTerminalAxisX), slot_x);
  const std::int32_t y = terminal_select_axis_slot_device(
      intervals_y, interval_count_y, selected_log_size,
      terminal_axis_mask_is_exact(selected_mask, kTerminalAxisY), slot_y);
  const std::int32_t z = terminal_select_axis_slot_device(
      intervals_z, interval_count_z, selected_log_size,
      terminal_axis_mask_is_exact(selected_mask, kTerminalAxisZ), slot_z);

  const std::uint32_t level = kMaxDepth - selected_log_size;
  return make_terminal_cell_request(
      {level, terminal_cell_prefix(x, y, z, level)});
}

__device__ inline std::uint32_t
terminal_leaf_box_count_device(const LeafBox &box) {
  if (box.x0 >= box.x1 || box.y0 >= box.y1 || box.z0 >= box.z1)
    return 0u;
  return static_cast<std::uint32_t>(box.x1 - box.x0) *
         static_cast<std::uint32_t>(box.y1 - box.y0) *
         static_cast<std::uint32_t>(box.z1 - box.z0);
}

__device__ inline std::int32_t terminal_min(std::int32_t lhs,
                                            std::int32_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

__device__ inline std::int32_t terminal_max(std::int32_t lhs,
                                            std::int32_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

__device__ inline bool select_terminal_partial_slab_device(
    const LeafBox &box, std::uint32_t &local_index, std::int32_t &leaf_x,
    std::int32_t &leaf_y, std::int32_t &leaf_z) {
  const std::uint32_t count = terminal_leaf_box_count_device(box);
  if (local_index >= count) {
    local_index -= count;
    return false;
  }

  const std::uint32_t count_x = static_cast<std::uint32_t>(box.x1 - box.x0);
  const std::uint32_t count_y = static_cast<std::uint32_t>(box.y1 - box.y0);
  const std::uint32_t plane_count = count_x * count_y;
  leaf_z = box.z0 + static_cast<std::int32_t>(local_index / plane_count);
  local_index -= (local_index / plane_count) * plane_count;
  leaf_y = box.y0 + static_cast<std::int32_t>(local_index / count_x);
  leaf_x = box.x0 + static_cast<std::int32_t>(local_index % count_x);
  return true;
}

__device__ inline void select_terminal_partial_leaf_device(
    const LeafBox &leaves, const LeafBox &full, std::uint32_t local_index,
    std::int32_t &leaf_x, std::int32_t &leaf_y, std::int32_t &leaf_z) {
  LeafBox core = leaves;
  if (select_terminal_partial_slab_device(
          {core.x0, terminal_min(full.x0, core.x1), core.y0, core.y1, core.z0,
           core.z1},
          local_index, leaf_x, leaf_y, leaf_z))
    return;
  core.x0 = terminal_min(terminal_max(full.x0, core.x0), core.x1);
  if (select_terminal_partial_slab_device(
          {terminal_max(full.x1, core.x0), core.x1, core.y0, core.y1, core.z0,
           core.z1},
          local_index, leaf_x, leaf_y, leaf_z))
    return;
  core.x1 = terminal_max(terminal_min(full.x1, core.x1), core.x0);

  if (select_terminal_partial_slab_device(
          {core.x0, core.x1, core.y0, terminal_min(full.y0, core.y1), core.z0,
           core.z1},
          local_index, leaf_x, leaf_y, leaf_z))
    return;
  core.y0 = terminal_min(terminal_max(full.y0, core.y0), core.y1);
  if (select_terminal_partial_slab_device(
          {core.x0, core.x1, terminal_max(full.y1, core.y0), core.y1, core.z0,
           core.z1},
          local_index, leaf_x, leaf_y, leaf_z))
    return;
  core.y1 = terminal_max(terminal_min(full.y1, core.y1), core.y0);

  if (select_terminal_partial_slab_device(
          {core.x0, core.x1, core.y0, core.y1, core.z0,
           terminal_min(full.z0, core.z1)},
          local_index, leaf_x, leaf_y, leaf_z))
    return;
  core.z0 = terminal_min(terminal_max(full.z0, core.z0), core.z1);
  if (select_terminal_partial_slab_device(
          {core.x0, core.x1, core.y0, core.y1, terminal_max(full.z1, core.z0),
           core.z1},
          local_index, leaf_x, leaf_y, leaf_z))
    return;
}

__device__ inline TerminalRequest
emit_terminal_node_full_request_device(const TerminalNodeInput &input,
                                       std::uint32_t local_index) {
  const TerminalBox box =
      terminal_box_device(input.depth, input.prefix, input.worldOffsetX,
                          input.worldOffsetY, input.worldOffsetZ);
  const LeafBox full = fully_covered_leaf_box_device(box);
  return select_terminal_full_request_device(full, local_index);
}

__device__ inline TerminalRequest
emit_terminal_node_leaf_request_device(const TerminalNodeInput &input,
                                       std::uint32_t local_index) {
  const TerminalBox box =
      terminal_box_device(input.depth, input.prefix, input.worldOffsetX,
                          input.worldOffsetY, input.worldOffsetZ);
  const LeafBox leaves = intersecting_leaf_box_device(box);
  const LeafBox full = fully_covered_leaf_box_device(box);
  std::int32_t leaf_x = 0;
  std::int32_t leaf_y = 0;
  std::int32_t leaf_z = 0;
  select_terminal_partial_leaf_device(leaves, full, local_index, leaf_x, leaf_y,
                                      leaf_z);
  const std::uint64_t mask =
      clipped_mask_for_leaf_device(box, leaf_x, leaf_y, leaf_z);
  return make_terminal_brick_request(
      {static_cast<std::uint32_t>(terminal_leaf_prefix(leaf_x, leaf_y, leaf_z)),
       mask});
}

__device__ inline TerminalRequest
emit_terminal_leaf_request_device(const TerminalLeafInput &input,
                                  std::uint32_t local_index) {
  std::uint32_t prefixes[kTerminalLeafBrickGroupCapacity]{};
  std::uint64_t masks[kTerminalLeafBrickGroupCapacity]{};
  std::uint32_t valid_mask = 0u;
  std::uint32_t leaf_x = 0u;
  std::uint32_t leaf_y = 0u;
  std::uint32_t leaf_z = 0u;
  if (!terminal_prefix_to_leaf_coord(kMaxDepth, input.leafPrefix, leaf_x,
                                     leaf_y, leaf_z)) {
    return {};
  }

  for (std::uint32_t bit = 0u; bit < 64u; ++bit) {
    if ((input.mask64 & (1ull << bit)) == 0u)
      continue;

    const std::int32_t local_x =
        static_cast<std::int32_t>(bit & (kLeafVoxelCount - 1u));
    const std::int32_t local_y = static_cast<std::int32_t>(
        (bit >> kLeafVoxelCountExp) & (kLeafVoxelCount - 1u));
    const std::int32_t local_z = static_cast<std::int32_t>(
        (bit >> (kLeafVoxelCountExp * 2u)) & (kLeafVoxelCount - 1u));
    const std::int32_t world_x =
        static_cast<std::int32_t>(leaf_x * kLeafVoxelCount) + local_x +
        input.worldOffsetX;
    const std::int32_t world_y =
        static_cast<std::int32_t>(leaf_y * kLeafVoxelCount) + local_y +
        input.worldOffsetY;
    const std::int32_t world_z =
        static_cast<std::int32_t>(leaf_z * kLeafVoxelCount) + local_z +
        input.worldOffsetZ;
    if (world_x < 0 || world_y < 0 || world_z < 0 ||
        world_x >= static_cast<std::int32_t>(kWorldVoxelCount) ||
        world_y >= static_cast<std::int32_t>(kWorldVoxelCount) ||
        world_z >= static_cast<std::int32_t>(kWorldVoxelCount)) {
      continue;
    }

    const std::int32_t out_leaf_x =
        world_x >> static_cast<std::int32_t>(kLeafVoxelCountExp);
    const std::int32_t out_leaf_y =
        world_y >> static_cast<std::int32_t>(kLeafVoxelCountExp);
    const std::int32_t out_leaf_z =
        world_z >> static_cast<std::int32_t>(kLeafVoxelCountExp);
    const std::uint32_t out_bit =
        static_cast<std::uint32_t>(world_x & (kLeafVoxelCount - 1u)) |
        (static_cast<std::uint32_t>(world_y & (kLeafVoxelCount - 1u))
         << kLeafVoxelCountExp) |
        (static_cast<std::uint32_t>(world_z & (kLeafVoxelCount - 1u))
         << (kLeafVoxelCountExp * 2u));
    const std::uint32_t prefix = static_cast<std::uint32_t>(
        terminal_leaf_prefix(out_leaf_x, out_leaf_y, out_leaf_z));
    std::uint32_t slot = 0u;
    bool found = false;
    for (; slot < kTerminalLeafBrickGroupCapacity; ++slot) {
      if (((valid_mask >> slot) & 1u) != 0u && prefixes[slot] == prefix) {
        found = true;
        break;
      }
    }

    if (!found) {
      slot = static_cast<std::uint32_t>(__popc(valid_mask));
      prefixes[slot] = prefix;
      valid_mask |= 1u << slot;
    }
    masks[slot] |= 1ull << out_bit;
  }

  return make_terminal_brick_request({prefixes[local_index],
                                      masks[local_index]});
}

__device__ inline std::uint32_t
find_request_owner_segment(const std::uint32_t *inclusive_offsets,
                           std::uint32_t segment_count,
                           std::uint32_t request_index) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = segment_count;
  const std::uint32_t target = request_index + 1u;
  while (lo < hi) {
    const std::uint32_t mid = lo + ((hi - lo) >> 1u);
    if (inclusive_offsets[mid] < target)
      lo = mid + 1u;
    else
      hi = mid;
  }
  return lo;
}

__device__ inline void store_terminal_emitted_request(
    std::uint32_t request_index, const TerminalRequest &request,
    TerminalRequest *requests, std::uint32_t *request_keys) {
  requests[request_index] = request;
  if (request_keys != nullptr) {
    request_keys[request_index] =
        terminal_request_representative_leaf_key(request);
  }
}

__global__ void emit_terminal_requests_kernel(
    const TerminalNodeInput *nodes, std::uint32_t node_count,
    const TerminalLeafInput *leaves, std::uint32_t leaf_count,
    const std::uint32_t *inclusive_offsets, std::uint32_t count_segment_count,
    std::uint32_t request_count, TerminalRequest *requests,
    std::uint32_t *request_keys = nullptr) {
  const std::uint32_t request_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (request_index >= request_count)
    return;

  const std::uint32_t segment_index = find_request_owner_segment(
      inclusive_offsets, count_segment_count, request_index);
  const std::uint32_t previous =
      segment_index == 0u ? 0u : inclusive_offsets[segment_index - 1u];
  const std::uint32_t local_index = request_index - previous;

  TerminalRequest request{};
  if (segment_index < node_count) {
    request =
        emit_terminal_node_full_request_device(nodes[segment_index],
                                               local_index);
  } else if (segment_index < node_count * 2u) {
    request = emit_terminal_node_leaf_request_device(
        nodes[segment_index - node_count], local_index);
  } else {
    request = emit_terminal_leaf_request_device(
        leaves[segment_index - node_count * 2u], local_index);
  }
  store_terminal_emitted_request(request_index, request, requests,
                                 request_keys);
}

} // namespace algo::svt::cuda::detail
