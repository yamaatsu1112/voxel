#pragma once

#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/terminal/detail/common.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::cuda::detail {

__device__ inline std::uint32_t
count_leaves_in_terminal_leaf_box_device(const LeafBox &box) {
  if (box.x0 >= box.x1 || box.y0 >= box.y1 || box.z0 >= box.z1)
    return 0u;

  return static_cast<std::uint32_t>(box.x1 - box.x0) *
         static_cast<std::uint32_t>(box.y1 - box.y0) *
         static_cast<std::uint32_t>(box.z1 - box.z0);
}

__device__ inline void
terminal_add_axis_dyadic_histogram(std::int32_t begin, std::int32_t end,
                                   std::uint32_t hist[kMaxDepth + 1u]) {
  std::uint32_t current = static_cast<std::uint32_t>(begin);
  const std::uint32_t limit = static_cast<std::uint32_t>(end);
  const std::uint32_t max_size = 1u << kMaxTerminalFullCellLogSize;
  while (current < limit) {
    const std::uint32_t remaining = limit - current;
    std::uint32_t aligned = current == 0u ? max_size : current & -current;
    if (aligned > max_size)
      aligned = max_size;
    std::uint32_t fit = 1u << (31u - __clz(remaining));
    std::uint32_t size = aligned < fit ? aligned : fit;
    const std::uint32_t log_size = 31u - __clz(size);
    ++hist[log_size];
    current += size;
  }
}

__device__ inline std::uint32_t
count_terminal_full_cell_requests_device(const LeafBox &full) {
  std::uint32_t suffix_x[kMaxDepth + 2u]{};
  std::uint32_t suffix_y[kMaxDepth + 2u]{};
  std::uint32_t suffix_z[kMaxDepth + 2u]{};
  terminal_add_axis_dyadic_histogram(full.x0, full.x1, suffix_x);
  terminal_add_axis_dyadic_histogram(full.y0, full.y1, suffix_y);
  terminal_add_axis_dyadic_histogram(full.z0, full.z1, suffix_z);
  terminal_axis_histogram_to_suffix_lengths_device(suffix_x);
  terminal_axis_histogram_to_suffix_lengths_device(suffix_y);
  terminal_axis_histogram_to_suffix_lengths_device(suffix_z);

  std::uint32_t count = 0u;
  for (std::uint32_t log_size = 0u;
       log_size <= kMaxTerminalFullCellLogSize; ++log_size) {
    count += terminal_full_count_for_log_size_device(suffix_x, suffix_y,
                                                     suffix_z, log_size);
  }
  return count;
}

__device__ inline std::uint32_t
count_terminal_partial_leaf_requests_device(const LeafBox &leaves,
                                            const LeafBox &full) {
  return count_leaves_in_terminal_leaf_box_device(leaves) -
         count_leaves_in_terminal_leaf_box_device(full);
}

struct TerminalNodeLeafBoxes {
  LeafBox leaves;
  LeafBox full;
};

__device__ inline TerminalNodeLeafBoxes
terminal_node_leaf_boxes_device(const TerminalNodeInput &input) {
  const TerminalBox box =
      terminal_box_device(input.depth, input.prefix, input.worldOffsetX,
                          input.worldOffsetY, input.worldOffsetZ);
  return {intersecting_leaf_box_device(box),
          fully_covered_leaf_box_device(box)};
}

__device__ inline std::uint32_t
count_terminal_leaf_requests_device(const TerminalLeafInput &input) {
  if (input.mask64 == 0u)
    return 0u;

  std::uint32_t leaf_x = 0u;
  std::uint32_t leaf_y = 0u;
  std::uint32_t leaf_z = 0u;
  if (!terminal_prefix_to_leaf_coord(kMaxDepth, input.leafPrefix, leaf_x,
                                     leaf_y, leaf_z)) {
    return 0u;
  }

  std::uint32_t leaf_mask = 0u;
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

    const std::uint32_t leaf_bit =
        ((static_cast<std::uint32_t>(world_x) >> kLeafVoxelCountExp) & 1u) |
        (((static_cast<std::uint32_t>(world_y) >> kLeafVoxelCountExp) & 1u)
         << 1u) |
        (((static_cast<std::uint32_t>(world_z) >> kLeafVoxelCountExp) & 1u)
         << 2u);
    leaf_mask |= 1u << leaf_bit;
  }
  return static_cast<std::uint32_t>(__popc(leaf_mask));
}

__global__ void count_terminal_requests_kernel(const TerminalNodeInput *nodes,
                                               std::uint32_t node_count,
                                               const TerminalLeafInput *leaves,
                                               std::uint32_t leaf_count,
                                               std::uint32_t *counts) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count_segment_count = node_count * 2u + leaf_count;
  if (index >= count_segment_count)
    return;

  if (index < node_count) {
    const TerminalNodeLeafBoxes boxes =
        terminal_node_leaf_boxes_device(nodes[index]);
    counts[index] = count_terminal_full_cell_requests_device(boxes.full);
  } else if (index < node_count * 2u) {
    const TerminalNodeLeafBoxes boxes =
        terminal_node_leaf_boxes_device(nodes[index - node_count]);
    counts[index] =
        count_terminal_partial_leaf_requests_device(boxes.leaves, boxes.full);
  } else {
    counts[index] =
        count_terminal_leaf_requests_device(leaves[index - node_count * 2u]);
  }
}

} // namespace algo::svt::cuda::detail
