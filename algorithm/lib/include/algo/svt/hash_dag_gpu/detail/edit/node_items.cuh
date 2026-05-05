#pragma once

#include <algo/svt/hash_dag_gpu/detail/edit_common.cuh>
#include <algo/svt/hash_dag_gpu/detail/node.cuh>
#include <algo/svt/hash_dag_gpu/types.cuh>

#include <cstdint>

namespace algo::svt::hash_dag_gpu::detail {

__global__ void build_level_node_items_kernel(
    const EditSvoRequest *requests, const std::uint32_t *request_count,
    std::uint32_t depth, const std::uint32_t *child_refs, NodeItem *node_items,
    std::uint32_t *item_words, std::uint32_t *temp_keys,
    std::uint32_t *representative_ids, std::uint32_t *canonical_refs) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= *request_count)
    return;
  if (request_depth(requests[index]) != depth)
    return;

  std::uint32_t children[kGroupSize]{};
  for (std::uint32_t child = 0u; child < kGroupSize; ++child)
    children[child] = child_refs[index * kGroupSize + child];

  // Repack the mutable child array into the canonical sparse node item used by
  // persistent hash tables.
  NodeItem item{};
  const std::uint32_t words = build_item_from_children(children, item.words);
  node_items[index] = item;
  item_words[index] = words;
  representative_ids[index] = index;

  const std::uint32_t child_mask = header_child_mask(item.words[0]);
  const std::uint32_t fill_mask = header_fill_mask(item.words[0]);
  if (child_mask == 0u) {
    // Uniform nodes do not need hash-table storage. They collapse to the same
    // sentinels used during traversal.
    if (fill_mask == 0u) {
      canonical_refs[index] = kEmptyRef;
      return;
    }
    if (fill_mask == kNodeMask) {
      canonical_refs[index] = kFullRef;
      return;
    }
  }

  // Temporary dedup keys include the item width so items with the same prefix
  // words but different logical lengths cannot collide.
  std::uint32_t *key = temp_keys + index * kTempDedupKeyWords;
  key[0] = words;
  for (std::uint32_t word = 0u; word < kMaxItemWords; ++word)
    key[word + 1u] = word < words ? item.words[word] : 0u;
}

} // namespace algo::svt::hash_dag_gpu::detail
