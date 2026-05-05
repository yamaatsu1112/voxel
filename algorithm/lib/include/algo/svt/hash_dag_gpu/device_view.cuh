#pragma once

#include <algo/cuda/hash_table/acceleration_hash/set/common.cuh>
#include <algo/cuda/hash_table/bump_allocator.cuh>
#include <algo/svt/hash_dag_gpu/config.cuh>

#include <cstdint>

namespace algo::svt::hash_dag_gpu {

// Plain device-side view of the DAG storage. HashDagGpu owns the allocations;
// kernels receive this trivially-copyable view so they can chase refs and
// canonicalize newly built nodes on device.
struct DeviceHashDagGpu {
  using Allocator = algo::cuda::BumpSlabAllocator;

  // Single root reference in device memory. Edits replace this after all child
  // refs have been rebuilt bottom-up.
  std::uint32_t *root_ref;
  // Separate tables by item width avoid storing variable-length items in one
  // allocation. A table ref carries this width in its high bits.
  algo::cuda::DeviceAccelerationHashSet32<1, Allocator> table1;
  algo::cuda::DeviceAccelerationHashSet32<2, Allocator> table2;
  algo::cuda::DeviceAccelerationHashSet32<3, Allocator> table3;
  algo::cuda::DeviceAccelerationHashSet32<4, Allocator> table4;
  algo::cuda::DeviceAccelerationHashSet32<5, Allocator> table5;
  algo::cuda::DeviceAccelerationHashSet32<6, Allocator> table6;
  algo::cuda::DeviceAccelerationHashSet32<7, Allocator> table7;
  algo::cuda::DeviceAccelerationHashSet32<8, Allocator> table8;
  algo::cuda::DeviceAccelerationHashSet32<9, Allocator> table9;
};

} // namespace algo::svt::hash_dag_gpu
