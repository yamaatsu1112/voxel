#pragma once

#include <algo/svt/constants.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace algo::svt {

struct [[gnu::packed]] SvtNode64 {
    uint32_t IsLeaf : 1;     // Indicates if this node is a leaf containing plain voxels.
    uint32_t ChildPtr : 31;  // Absolute offset to array of existing child nodes/voxels.
    uint64_t ChildMask;      // Indicates which children/voxels are present in array.
};

static_assert(sizeof(SvtNode64) == 12);
static_assert(std::is_trivially_copyable_v<SvtNode64>);

class SVO64GroupedNoLeaf {
  public:
    explicit SVO64GroupedNoLeaf(uint32_t max_depth = DEFAULT_MAX_DEPTH)
        : max_depth_(max_depth) {
        validate_max_depth();
        nodes_.push_back(make_node(false));
    }

    void set_voxel(uint32_t x, uint32_t y, uint32_t z, bool value) {
        validate_coordinates(x, y, z);

        uint32_t node_index = kRootNodeIndex;
        std::vector<uint32_t> node_indices(max_depth_ + 1);
        std::vector<uint32_t> child_indices(max_depth_);
        node_indices[0] = node_index;

        for (uint32_t depth = 0; depth < max_depth_; ++depth) {
            const uint32_t child_index = get_child_index(x, y, z, depth);
            child_indices[depth] = child_index;

            Node& node = nodes_[node_index];
            if (node.IsLeaf != 0) {
                throw std::logic_error(
                    "SVO64GroupedNoLeaf: internal traversal reached leaf node");
            }
            if (has_child(node, child_index)) {
                node_index = get_child_node_index(node, child_index);
                node_indices[depth + 1] = node_index;
                continue;
            }

            if (!value) {
                return;
            }

            const uint32_t new_node_index =
                allocate_node(depth + 1 == max_depth_);
            insert_child(node_index, child_index, new_node_index);
            node_index = new_node_index;
            node_indices[depth + 1] = node_index;
        }

        Node& leaf = nodes_[node_index];
        if (leaf.IsLeaf == 0) {
            throw std::logic_error(
                "SVO64GroupedNoLeaf: terminal node is not marked as leaf");
        }
        const uint64_t voxel_bit = uint64_t{1} << get_leaf_offset(x, y, z);
        const bool currently_filled = (leaf.ChildMask & voxel_bit) != 0;
        if (currently_filled == value) {
            return;
        }

        if (value) {
            leaf.ChildMask |= voxel_bit;
        } else {
            leaf.ChildMask &= ~voxel_bit;
        }

        collapse_empty_path(node_indices, child_indices);
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        uint32_t node_index = kRootNodeIndex;
        for (uint32_t depth = 0; depth < max_depth_; ++depth) {
            const Node& node = nodes_[node_index];
            if (node.IsLeaf != 0) {
                throw std::logic_error(
                    "SVO64GroupedNoLeaf: internal traversal reached leaf node");
            }
            const uint32_t child_index = get_child_index(x, y, z, depth);
            if (!has_child(node, child_index)) {
                return false;
            }
            node_index = get_child_node_index(node, child_index);
        }

        const Node& leaf = nodes_[node_index];
        if (leaf.IsLeaf == 0) {
            throw std::logic_error(
                "SVO64GroupedNoLeaf: terminal node is not marked as leaf");
        }
        const uint64_t voxel_bit = uint64_t{1} << get_leaf_offset(x, y, z);
        return (leaf.ChildMask & voxel_bit) != 0;
    }

    [[nodiscard]] std::size_t node_count() const {
        return nodes_.size() - free_node_indices_.size();
    }

    [[nodiscard]] std::size_t leaf_count() const {
        return 0;
    }

    [[nodiscard]] uint32_t world_voxel_count() const {
        return world_voxel_count(max_depth_);
    }

    [[nodiscard]] uint32_t max_depth() const {
        return max_depth_;
    }

    [[nodiscard]] static constexpr uint32_t world_voxel_count(
        uint32_t max_depth) {
        return LEAF_VOXEL_COUNT << (max_depth * kAxisBitsPerLevel);
    }

    [[nodiscard]] static constexpr uint32_t
    max_depth_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(
            requested_world_size, LEAF_VOXEL_COUNT, kAxisBitsPerLevel, 1,
            kMaxSupportedDepth);
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        std::size_t free_child_block_bytes = 0;
        for (const auto& offsets : free_child_block_offsets_) {
            free_child_block_bytes += offsets.capacity() * sizeof(uint32_t);
        }

        return nodes_.capacity() * sizeof(Node) +
               child_indices_.capacity() * sizeof(uint32_t) +
               free_node_indices_.capacity() * sizeof(uint32_t) +
               free_child_block_bytes;
    }

    [[nodiscard]] std::size_t node_storage_bytes() const {
        std::size_t child_block_bytes = 0;
        for (const auto& node : nodes_) {
            child_block_bytes += std::popcount(node.ChildMask) * sizeof(uint32_t);
        }

        return node_count() * sizeof(Node) + child_block_bytes;
    }

  private:
    using Node = SvtNode64;

    static constexpr uint32_t kAxisBitsPerLevel = 2;
    static constexpr uint32_t kAxisChildCount = 1u << kAxisBitsPerLevel;
    static constexpr uint32_t kChildCount =
        kAxisChildCount * kAxisChildCount * kAxisChildCount;
    static constexpr uint32_t kRootNodeIndex = 0;
    static constexpr uint32_t kMaxSupportedDepth = 14;
    static constexpr uint32_t kMaxChildPtr = 0x7FFFFFFFu;

    [[nodiscard]] static Node make_node(bool is_leaf) {
        Node node{};
        node.IsLeaf = is_leaf ? 1u : 0u;
        node.ChildPtr = 0;
        node.ChildMask = 0;
        return node;
    }

    [[nodiscard]] static bool has_child(const Node& node, uint32_t child_index) {
        return (node.ChildMask & (uint64_t{1} << child_index)) != 0;
    }

    [[nodiscard]] static bool is_empty(const Node& node) {
        return node.ChildMask == 0;
    }

    [[nodiscard]] static uint32_t child_rank(uint64_t child_mask,
                                             uint32_t child_index) {
        if (child_index == 0) {
            return 0;
        }

        const uint64_t lower_bits = (uint64_t{1} << child_index) - 1;
        return std::popcount(child_mask & lower_bits);
    }

    [[nodiscard]] uint32_t allocate_node(bool is_leaf) {
        if (!free_node_indices_.empty()) {
            const uint32_t index = free_node_indices_.back();
            free_node_indices_.pop_back();
            nodes_[index] = make_node(is_leaf);
            return index;
        }

        const auto index = static_cast<uint32_t>(nodes_.size());
        if (nodes_.size() >= static_cast<std::size_t>(UINT32_MAX)) {
            throw std::overflow_error("SVO64GroupedNoLeaf: node index overflow");
        }
        nodes_.push_back(make_node(is_leaf));
        return index;
    }

    void free_node(uint32_t index) {
        if (index == kRootNodeIndex || index >= nodes_.size()) {
            throw std::out_of_range("SVO64GroupedNoLeaf: node index out of bounds");
        }
        free_node_indices_.push_back(index);
    }

    [[nodiscard]] uint32_t allocate_child_block(uint32_t size) {
        if (size == 0) {
            return 0;
        }

        auto& free_blocks = free_child_block_offsets_[size];
        if (!free_blocks.empty()) {
            const uint32_t offset = free_blocks.back();
            free_blocks.pop_back();
            return offset;
        }

        const std::size_t offset = child_indices_.size();
        if (offset > static_cast<std::size_t>(kMaxChildPtr) ||
            size > static_cast<std::size_t>(kMaxChildPtr) - offset + 1) {
            throw std::overflow_error(
                "SVO64GroupedNoLeaf: child block offset overflow");
        }

        child_indices_.resize(offset + size);
        return static_cast<uint32_t>(offset);
    }

    void free_child_block(uint32_t offset, uint32_t size) {
        if (size == 0) {
            return;
        }
        if (offset + size > child_indices_.size()) {
            throw std::out_of_range(
                "SVO64GroupedNoLeaf: child block index out of bounds");
        }
        free_child_block_offsets_[size].push_back(offset);
    }

    [[nodiscard]] uint32_t get_child_node_index(const Node& node,
                                                uint32_t child_index) const {
        const uint32_t rank = child_rank(node.ChildMask, child_index);
        const std::size_t offset = static_cast<std::size_t>(node.ChildPtr) + rank;
        if (offset >= child_indices_.size()) {
            throw std::out_of_range(
                "SVO64GroupedNoLeaf: child node index out of bounds");
        }
        return child_indices_[offset];
    }

    void insert_child(uint32_t parent_index,
                      uint32_t child_index,
                      uint32_t new_child_index) {
        Node& parent = nodes_[parent_index];
        const uint64_t old_mask = parent.ChildMask;
        const uint32_t old_count = std::popcount(old_mask);
        const uint32_t new_count = old_count + 1;
        const uint32_t insert_rank = child_rank(old_mask, child_index);
        const uint32_t new_offset = allocate_child_block(new_count);

        for (uint32_t i = 0; i < insert_rank; ++i) {
            child_indices_[new_offset + i] = child_indices_[parent.ChildPtr + i];
        }

        child_indices_[new_offset + insert_rank] = new_child_index;

        for (uint32_t i = insert_rank; i < old_count; ++i) {
            child_indices_[new_offset + i + 1] = child_indices_[parent.ChildPtr + i];
        }

        free_child_block(parent.ChildPtr, old_count);
        parent.ChildPtr = new_offset;
        parent.ChildMask = old_mask | (uint64_t{1} << child_index);
    }

    void remove_child(uint32_t parent_index, uint32_t child_index) {
        Node& parent = nodes_[parent_index];
        const uint64_t old_mask = parent.ChildMask;
        const uint32_t old_count = std::popcount(old_mask);
        const uint32_t remove_rank = child_rank(old_mask, child_index);
        const uint64_t new_mask = old_mask & ~(uint64_t{1} << child_index);
        const uint32_t new_count = old_count - 1;
        const uint32_t old_offset = parent.ChildPtr;
        const uint32_t new_offset =
            new_count == 0 ? 0 : allocate_child_block(new_count);

        for (uint32_t i = 0; i < remove_rank; ++i) {
            child_indices_[new_offset + i] = child_indices_[old_offset + i];
        }

        for (uint32_t i = remove_rank + 1; i < old_count; ++i) {
            child_indices_[new_offset + i - 1] = child_indices_[old_offset + i];
        }

        free_child_block(old_offset, old_count);
        parent.ChildPtr = new_offset;
        parent.ChildMask = new_mask;
    }

    [[nodiscard]] uint32_t get_child_index(uint32_t x,
                                           uint32_t y,
                                           uint32_t z,
                                           uint32_t depth) const {
        const uint32_t shift = (max_depth_ - depth - 1) * kAxisBitsPerLevel;
        const uint32_t scale = LEAF_VOXEL_COUNT << shift;
        const uint32_t x_digit = (x / scale) & (kAxisChildCount - 1);
        const uint32_t y_digit = (y / scale) & (kAxisChildCount - 1);
        const uint32_t z_digit = (z / scale) & (kAxisChildCount - 1);
        return x_digit | (y_digit << kAxisBitsPerLevel) |
               (z_digit << (kAxisBitsPerLevel * 2));
    }

    [[nodiscard]] static uint32_t get_leaf_offset(uint32_t x,
                                                  uint32_t y,
                                                  uint32_t z) {
        const uint32_t x_offset = x % LEAF_VOXEL_COUNT;
        const uint32_t y_offset = y % LEAF_VOXEL_COUNT;
        const uint32_t z_offset = z % LEAF_VOXEL_COUNT;
        return x_offset + y_offset * LEAF_VOXEL_COUNT +
               z_offset * LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT;
    }

    void validate_coordinates(uint32_t x, uint32_t y, uint32_t z) const {
        if (x >= world_voxel_count() || y >= world_voxel_count() ||
            z >= world_voxel_count()) {
            throw std::out_of_range(
                "SVO64GroupedNoLeaf: voxel coordinates out of bounds");
        }
    }

    void validate_max_depth() const {
        if (max_depth_ == 0 || max_depth_ > kMaxSupportedDepth) {
            throw std::out_of_range(
                "SVO64GroupedNoLeaf: max_depth out of supported range");
        }
    }

    void collapse_empty_path(const std::vector<uint32_t>& node_indices,
                             const std::vector<uint32_t>& child_indices) {
        for (std::size_t depth = max_depth_; depth > 0; --depth) {
            const uint32_t current_index = node_indices[depth];
            if (!is_empty(nodes_[current_index])) {
                break;
            }

            remove_child(node_indices[depth - 1], child_indices[depth - 1]);
            free_node(current_index);
        }
    }

    std::vector<Node> nodes_;
    std::vector<uint32_t> child_indices_;
    std::vector<uint32_t> free_node_indices_;
    std::array<std::vector<uint32_t>, kChildCount + 1> free_child_block_offsets_;
    uint32_t max_depth_;
};

} // namespace algo::svt
