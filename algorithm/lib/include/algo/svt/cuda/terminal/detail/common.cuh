#pragma once

#include <algo/svt/cuda/config.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/terminal/detail/boxes.cuh>
#include <algo/svt/cuda/terminal/types.cuh>

#include <cstdint>

namespace algo::svt::cuda::detail {

inline constexpr std::uint32_t kMaxTerminalFullCellLogSize = kMaxDepth - 1u;

__device__ inline std::int32_t terminal_floor_div(std::int32_t value,
                                                  std::int32_t divisor) {
  std::int32_t quotient = value / divisor;
  const std::int32_t remainder = value % divisor;
  if (remainder != 0 && ((remainder < 0) != (divisor < 0)))
    --quotient;
  return quotient;
}

__device__ inline std::int32_t terminal_ceil_div(std::int32_t value,
                                                 std::int32_t divisor) {
  return -terminal_floor_div(-value, divisor);
}

__device__ inline std::uint32_t terminal_child_x(std::uint32_t child) {
  return child & 1u;
}

__device__ inline std::uint32_t terminal_child_y(std::uint32_t child) {
  return (child >> 1u) & 1u;
}

__device__ inline std::uint32_t terminal_child_z(std::uint32_t child) {
  return (child >> 2u) & 1u;
}

__device__ inline void
terminal_axis_histogram_to_suffix_lengths_device(
    std::uint32_t suffix[kMaxDepth + 2u]) {
  suffix[kMaxDepth + 1u] = 0u;
  for (std::int32_t index = static_cast<std::int32_t>(kMaxDepth); index >= 0;
       --index) {
    const std::uint32_t log_size = static_cast<std::uint32_t>(index);
    suffix[log_size] = suffix[log_size + 1u] + (suffix[log_size] << log_size);
  }
}

__device__ inline std::uint32_t terminal_full_count_for_log_size_device(
    const std::uint32_t suffix_x[kMaxDepth + 2u],
    const std::uint32_t suffix_y[kMaxDepth + 2u],
    const std::uint32_t suffix_z[kMaxDepth + 2u], std::uint32_t log_size) {
  const std::uint32_t current =
      suffix_x[log_size] * suffix_y[log_size] * suffix_z[log_size];
  const std::uint32_t next = suffix_x[log_size + 1u] *
                             suffix_y[log_size + 1u] *
                             suffix_z[log_size + 1u];
  return (current - next) >> (kGroupSizeExp * log_size);
}

__device__ inline bool terminal_prefix_to_leaf_coord(std::uint32_t depth,
                                                     std::uint32_t prefix,
                                                     std::uint32_t &leaf_x,
                                                     std::uint32_t &leaf_y,
                                                     std::uint32_t &leaf_z) {
  if (depth > kMaxDepth)
    return false;
  if (depth < 11u && (prefix >> (depth * kGroupSizeExp)) != 0u)
    return false;

  leaf_x = 0u;
  leaf_y = 0u;
  leaf_z = 0u;
  for (std::uint32_t i = 0u; i < depth; ++i) {
    const std::uint32_t shift = (depth - i - 1u) * kGroupSizeExp;
    const std::uint32_t child =
        static_cast<std::uint32_t>((prefix >> shift) & (kGroupSize - 1u));
    leaf_x = (leaf_x << 1u) | terminal_child_x(child);
    leaf_y = (leaf_y << 1u) | terminal_child_y(child);
    leaf_z = (leaf_z << 1u) | terminal_child_z(child);
  }
  const std::uint32_t remaining = kMaxDepth - depth;
  leaf_x <<= remaining;
  leaf_y <<= remaining;
  leaf_z <<= remaining;
  return true;
}

__device__ inline TerminalBox terminal_box_device(std::uint32_t depth,
                                                  std::uint32_t prefix,
                                                  std::int32_t offset_x,
                                                  std::int32_t offset_y,
                                                  std::int32_t offset_z) {
  std::uint32_t leaf_x = 0u;
  std::uint32_t leaf_y = 0u;
  std::uint32_t leaf_z = 0u;
  if (!terminal_prefix_to_leaf_coord(depth, prefix, leaf_x, leaf_y, leaf_z))
    return {};

  const std::int32_t size =
      static_cast<std::int32_t>(kLeafVoxelCount << (kMaxDepth - depth));
  const std::int32_t x0 =
      static_cast<std::int32_t>(leaf_x * kLeafVoxelCount) + offset_x;
  const std::int32_t y0 =
      static_cast<std::int32_t>(leaf_y * kLeafVoxelCount) + offset_y;
  const std::int32_t z0 =
      static_cast<std::int32_t>(leaf_z * kLeafVoxelCount) + offset_z;
  return {x0, y0, z0, size};
}

__device__ inline LeafBox intersect_world_leaves_device(LeafBox box) {
  box.x0 = box.x0 < 0 ? 0 : box.x0;
  box.y0 = box.y0 < 0 ? 0 : box.y0;
  box.z0 = box.z0 < 0 ? 0 : box.z0;
  box.x1 = box.x1 > static_cast<std::int32_t>(kLeafCountPerAxis)
               ? static_cast<std::int32_t>(kLeafCountPerAxis)
               : box.x1;
  box.y1 = box.y1 > static_cast<std::int32_t>(kLeafCountPerAxis)
               ? static_cast<std::int32_t>(kLeafCountPerAxis)
               : box.y1;
  box.z1 = box.z1 > static_cast<std::int32_t>(kLeafCountPerAxis)
               ? static_cast<std::int32_t>(kLeafCountPerAxis)
               : box.z1;
  return box;
}

__device__ inline LeafBox intersecting_leaf_box_device(const TerminalBox &box) {
  const std::int32_t x1 = box.x + box.size;
  const std::int32_t y1 = box.y + box.size;
  const std::int32_t z1 = box.z + box.size;
  return intersect_world_leaves_device(
      {terminal_floor_div(box.x, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_ceil_div(x1, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_floor_div(box.y, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_ceil_div(y1, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_floor_div(box.z, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_ceil_div(z1, static_cast<std::int32_t>(kLeafVoxelCount))});
}

__device__ inline LeafBox
fully_covered_leaf_box_device(const TerminalBox &box) {
  const std::int32_t x1 = box.x + box.size;
  const std::int32_t y1 = box.y + box.size;
  const std::int32_t z1 = box.z + box.size;
  return intersect_world_leaves_device(
      {terminal_ceil_div(box.x, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_floor_div(x1, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_ceil_div(box.y, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_floor_div(y1, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_ceil_div(box.z, static_cast<std::int32_t>(kLeafVoxelCount)),
       terminal_floor_div(z1, static_cast<std::int32_t>(kLeafVoxelCount))});
}

__device__ inline std::uint32_t terminal_leaf_prefix(std::int32_t leaf_x,
                                                     std::int32_t leaf_y,
                                                     std::int32_t leaf_z) {
  return make_leaf_key(static_cast<std::uint32_t>(leaf_x),
                       static_cast<std::uint32_t>(leaf_y),
                       static_cast<std::uint32_t>(leaf_z));
}

__device__ inline std::uint32_t terminal_cell_prefix(std::int32_t leaf_x,
                                                     std::int32_t leaf_y,
                                                     std::int32_t leaf_z,
                                                     std::uint32_t level) {
  return prefix_for_leaf_key(make_leaf_key(static_cast<std::uint32_t>(leaf_x),
                                           static_cast<std::uint32_t>(leaf_y),
                                           static_cast<std::uint32_t>(leaf_z)),
                             level);
}

__device__ inline std::uint32_t
terminal_request_representative_leaf_key(const TerminalRequest &request) {
  if (terminal_request_is_brick(request))
    return terminal_request_brick(request).leafPrefix;

  const CellWriteRequest cell = terminal_request_cell(request);
  const std::uint32_t shift =
      (kMaxDepth - cell.level) * kGroupSizeExp;
  return static_cast<std::uint32_t>(cell.prefix << shift);
}

__device__ inline std::uint64_t
clipped_mask_for_leaf_device(const TerminalBox &box, std::int32_t leaf_x,
                             std::int32_t leaf_y, std::int32_t leaf_z) {
  std::uint64_t mask = 0u;
  const std::int32_t x1 = box.x + box.size;
  const std::int32_t y1 = box.y + box.size;
  const std::int32_t z1 = box.z + box.size;
  const std::int32_t base_x =
      leaf_x * static_cast<std::int32_t>(kLeafVoxelCount);
  const std::int32_t base_y =
      leaf_y * static_cast<std::int32_t>(kLeafVoxelCount);
  const std::int32_t base_z =
      leaf_z * static_cast<std::int32_t>(kLeafVoxelCount);
  for (std::uint32_t z = 0u; z < kLeafVoxelCount; ++z)
    for (std::uint32_t y = 0u; y < kLeafVoxelCount; ++y)
      for (std::uint32_t x = 0u; x < kLeafVoxelCount; ++x) {
        const std::int32_t world_x = base_x + static_cast<std::int32_t>(x);
        const std::int32_t world_y = base_y + static_cast<std::int32_t>(y);
        const std::int32_t world_z = base_z + static_cast<std::int32_t>(z);
        if (world_x >= box.x && world_x < x1 && world_y >= box.y &&
            world_y < y1 && world_z >= box.z && world_z < z1) {
          const std::uint32_t bit =
              x | (y << kLeafVoxelCountExp) | (z << (kLeafVoxelCountExp * 2u));
          mask |= 1ull << bit;
        }
      }
  return mask;
}

} // namespace algo::svt::cuda::detail
