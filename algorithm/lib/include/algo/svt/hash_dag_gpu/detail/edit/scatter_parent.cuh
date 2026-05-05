#pragma once

#include <algo/svt/hash_dag_gpu/config.cuh>
#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

__global__ void scatter_level_to_parent_kernel(
    const EditSvoRequest *requests, const std::uint32_t *request_count,
    std::uint32_t depth, const std::uint32_t *canonical_refs,
    std::uint32_t *child_refs) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= *request_count)
    return;
  const EditSvoRequest request = requests[index];
  if (request_depth(request) != depth)
    return;

  const std::uint32_t parent = request.parent_request;
  if (parent == kInvalidRequest)
    return;
  // The low 3 bits of a node prefix identify the child slot within its parent.
  // Parent child_refs already contain unchanged siblings from initialization.
  const std::uint32_t child = request.prefix & (kGroupSize - 1u);
  child_refs[parent * kGroupSize + child] = canonical_refs[index];
}

} // namespace algo::svt::hash_dag_gpu::detail
