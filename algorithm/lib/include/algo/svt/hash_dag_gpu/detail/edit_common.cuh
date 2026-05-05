#pragma once

#include <algo/cuda/acceleration_hash.cuh>
#include <algo/svt/hash_dag_gpu/config.cuh>
#include <algo/svt/hash_dag_gpu/detail/node.cuh>
#include <algo/svt/hash_dag_gpu/device_view.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

inline constexpr std::uint32_t kKernelBlockSize = 256u;
// Edit requests exist for every touched leaf plus each ancestor level,
// including depth 0 for the root and kInnerDepth for leaves.
inline constexpr std::uint32_t kEditDepthCount = kInnerDepth + 1u;
inline constexpr std::uint32_t kInvalidRequest = 0xffffffffu;
inline constexpr std::uint32_t kTempDedupKeyWords = kMaxItemWords + 1u;

enum EditStatus : std::uint32_t {
  kEditStatusOk = 0u,
  kEditStatusHashOverflow = 1u,
  kEditStatusWorkspaceOverflow = 2u,
};

struct EditSvoRequest {
  // One node in the temporary SVO-shaped edit request tree built from a batch
  // of leaf masks before those edits are folded back into the hash DAG.
  // Depth of the node represented by this request. 0 is root; kInnerDepth is a
  // leaf. Stored as a plain integer despite the name for compactness.
  std::uint32_t packed_depth;
  // Path prefix to this node, packed as 3-bit child indices from the root.
  std::uint32_t prefix;
  // Index of the parent request in the compacted request array, or
  // kInvalidRequest for the root.
  std::uint32_t parent_request;
  // Reserved for future per-request flags; kept in the struct so layout changes
  // are deliberate.
  std::uint32_t metadata;
  // Index into leaf_masks for leaf requests. Ancestor requests carry the
  // first leaf-mask slot that emitted the prefix, but leaf update is the only
  // phase that reads this field.
  std::uint32_t source_index;
};

static_assert(sizeof(EditSvoRequest) == sizeof(std::uint32_t) * 5u);

using TempDedupMap =
    algo::cuda::DeviceAccelerationHashMap32<kTempDedupKeyWords,
                                            DeviceHashDagGpu::Allocator>;
using TempDedupSlab = typename TempDedupMap::Slab;

inline std::uint32_t ceil_div_u32(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1u) / divisor;
}

inline std::uint32_t block_count(std::uint32_t count,
                                 std::uint32_t block_size) {
  return (count + block_size - 1u) / block_size;
}

inline cudaError_t last_launch_status() {
  const cudaError_t status = cudaGetLastError();
  return status == cudaSuccess ? cudaSuccess : status;
}

inline std::uint32_t max_request_count(std::uint32_t count) {
  return count * kEditDepthCount;
}

inline std::uint32_t request_offset_count(std::uint32_t count) {
  return count;
}

__host__ __device__ inline std::uint32_t
prefix_for_leaf_key(std::uint32_t leaf_key, std::uint32_t depth) {
  // Truncate a leaf key to the prefix that identifies its ancestor at depth.
  if (depth == 0u)
    return 0u;
  return leaf_key >> ((kInnerDepth - depth) * kGroupSizeExp);
}

__host__ __device__ inline std::uint32_t
child_index_from_prefix(std::uint32_t prefix, std::uint32_t depth,
                        std::uint32_t level) {
  // Extract the child taken at 'level' while walking from root to a node at
  // 'depth'. level is strictly above depth.
  const std::uint32_t shift = (depth - level - 1u) * kGroupSizeExp;
  return (prefix >> shift) & (kGroupSize - 1u);
}

#ifdef __CUDACC__

__device__ inline void set_edit_status(std::uint32_t *status,
                                       std::uint32_t value) {
  atomicCAS(status, kEditStatusOk, value);
}

__device__ inline std::uint32_t request_depth(const EditSvoRequest &request) {
  return request.packed_depth;
}

__device__ inline bool is_request_non_uniform(const NodeItem *items,
                                              std::uint32_t request_index) {
  const std::uint32_t header = items[request_index].words[0];
  const std::uint32_t child_mask = header_child_mask(header);
  const std::uint32_t fill_mask = header_fill_mask(header);
  return !(child_mask == 0u && (fill_mask == 0u || fill_mask == kNodeMask));
}

#endif

} // namespace algo::svt::hash_dag_gpu::detail
