#pragma once

#include <algo/svt/cuda/config.cuh>

#include <cstdint>

namespace algo::svt::cuda {

// Internal nodes store eight child slots. Each slot is encoded with the bit
// layout documented in config.cuh.
struct GpuSvoNode {
    std::uint32_t child_data[kGroupSize];
};

// A leaf stores the 4^3 voxel payload as two 32-bit masks. Bit index is computed
// by leaf_offset_for_voxel(), with the low mask covering offsets [0, 31].
struct GpuSvoLeaf {
    std::uint32_t voxel_data_low;
    std::uint32_t voxel_data_high;
};

// Counters live on the device so kernels can allocate and recycle nodes/leaves.
// free_*_count is the stack size for the matching free_*_indices array.
struct GpuSvoCounters {
    std::uint32_t node_count;
    std::uint32_t leaf_count;
    std::uint32_t free_node_count;
    std::uint32_t free_leaf_count;
};

// Non-owning device-side view passed to kernels. GpuSvo in storage.cuh owns the
// allocations and materializes this view for launches.
struct DeviceGpuSvo {
    GpuSvoNode* nodes;
    GpuSvoLeaf* leaves;
    GpuSvoCounters* counters;
    std::uint32_t* free_node_indices;
    std::uint32_t* free_leaf_indices;
    std::uint32_t max_node_count;
    std::uint32_t max_leaf_count;
};

static_assert(sizeof(GpuSvoNode) == sizeof(std::uint32_t) * kGroupSize);
static_assert(sizeof(GpuSvoLeaf) == 8);

} // namespace algo::svt::cuda
