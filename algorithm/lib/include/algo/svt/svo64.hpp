#pragma once

#include <algo/svt/constants.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace algo::svt {

class SVO64 {
  public:
    explicit SVO64(uint32_t max_depth = DEFAULT_MAX_DEPTH) : max_depth_(max_depth) {
        validate_max_depth();
        nodes_.push_back(Node{});
    }

    void set_voxel(uint32_t x, uint32_t y, uint32_t z, bool value) {
        validate_coordinates(x, y, z);

        uint32_t node_index = kRootNodeIndex;
        std::vector<uint32_t> node_indices(max_depth_);
        std::vector<uint32_t> child_indices(max_depth_);

        for (uint32_t depth = 0; depth + 1 < max_depth_; ++depth) {
            const uint32_t child_index = get_child_index(x, y, z, depth);
            node_indices[depth] = node_index;
            child_indices[depth] = child_index;

            if (nodes_[node_index].has_child(child_index)) {
                node_index = nodes_[node_index].get_child_index(child_index);
                continue;
            }

            if (nodes_[node_index].is_filled(child_index) == value) {
                return;
            }

            const bool filled = nodes_[node_index].is_filled(child_index);
            const uint32_t new_node_index = allocate_node();
            Node& node = nodes_[node_index];
            node.set_child_mask(child_index, true);
            node.set_child_index(child_index, new_node_index);

            Node& new_node = nodes_[new_node_index];
            new_node.filled_mask = filled ? kLeafFullMask : 0;
            node_index = new_node_index;
        }

        const uint32_t child_index = get_child_index(x, y, z, max_depth_ - 1);
        node_indices[max_depth_ - 1] = node_index;
        child_indices[max_depth_ - 1] = child_index;

        uint32_t leaf_index;
        if (nodes_[node_index].has_child(child_index)) {
            leaf_index = nodes_[node_index].get_child_index(child_index);
        } else {
            if (nodes_[node_index].is_filled(child_index) == value) {
                return;
            }

            const bool filled = nodes_[node_index].is_filled(child_index);
            leaf_index = allocate_leaf();
            Node& node = nodes_[node_index];
            node.set_child_mask(child_index, true);
            node.set_child_index(child_index, leaf_index);
            leaves_[leaf_index].voxel_data = filled ? kLeafFullMask : 0;
        }

        Leaf& leaf = leaves_[leaf_index];
        const uint32_t bit_offset = get_leaf_offset(x, y, z);
        const uint64_t mask = uint64_t{1} << bit_offset;

        if (value) {
            leaf.voxel_data |= mask;
        } else {
            leaf.voxel_data &= ~mask;
        }

        free_nodes_to_be_freed(node_indices, child_indices, leaf_index);
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        uint32_t node_index = kRootNodeIndex;
        for (uint32_t depth = 0; depth + 1 < max_depth_; ++depth) {
            const Node& node = nodes_[node_index];
            const uint32_t child_index = get_child_index(x, y, z, depth);

            if (!node.has_child(child_index)) {
                return node.is_filled(child_index);
            }
            node_index = node.get_child_index(child_index);
        }

        const Node& node = nodes_[node_index];
        const uint32_t child_index = get_child_index(x, y, z, max_depth_ - 1);
        if (!node.has_child(child_index)) {
            return node.is_filled(child_index);
        }

        const Leaf& leaf = leaves_[node.get_child_index(child_index)];
        const uint64_t mask = uint64_t{1} << get_leaf_offset(x, y, z);
        return (leaf.voxel_data & mask) != 0;
    }

    [[nodiscard]] std::size_t node_count() const {
        return nodes_.size() - free_node_indices_.size();
    }

    [[nodiscard]] std::size_t leaf_count() const {
        return leaves_.size() - free_leaf_indices_.size();
    }

    [[nodiscard]] uint32_t world_voxel_count() const {
        return world_voxel_count(max_depth_);
    }

    [[nodiscard]] uint32_t max_depth() const {
        return max_depth_;
    }

    [[nodiscard]] static constexpr uint32_t world_voxel_count(uint32_t max_depth) {
        return LEAF_VOXEL_COUNT << (max_depth * kAxisBitsPerLevel);
    }

    [[nodiscard]] static constexpr uint32_t
    max_depth_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(
            requested_world_size, LEAF_VOXEL_COUNT, kAxisBitsPerLevel, 1,
            kMaxSupportedDepth);
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        return nodes_.capacity() * sizeof(Node) +
               leaves_.capacity() * sizeof(Leaf) +
               free_node_indices_.capacity() * sizeof(uint32_t) +
               free_leaf_indices_.capacity() * sizeof(uint32_t);
    }

    [[nodiscard]] std::size_t node_storage_bytes() const {
        return node_count() * sizeof(Node) + leaf_count() * sizeof(Leaf);
    }

  private:
    static constexpr uint32_t kAxisBitsPerLevel = 2;
    static constexpr uint32_t kAxisChildCount = 1u << kAxisBitsPerLevel;
    static constexpr uint32_t kChildCount =
        kAxisChildCount * kAxisChildCount * kAxisChildCount;
    static constexpr uint32_t kRootNodeIndex = 0;
    static constexpr uint64_t kLeafFullMask = ~uint64_t{0};
    static constexpr uint32_t kMaxSupportedDepth = 14;

    struct Node {
        uint64_t child_mask = 0;
        uint64_t filled_mask = 0;
        uint32_t child_indices[kChildCount] = {};

        [[nodiscard]] bool has_child(uint32_t child_index) const {
            return (child_mask & (uint64_t{1} << child_index)) != 0;
        }

        [[nodiscard]] bool is_filled(uint32_t child_index) const {
            return (filled_mask & (uint64_t{1} << child_index)) != 0;
        }

        [[nodiscard]] uint32_t get_child_index(uint32_t child_index) const {
            return child_indices[child_index];
        }

        void set_child_mask(uint32_t child_index, bool value) {
            const uint64_t bit = uint64_t{1} << child_index;
            if (value) {
                child_mask |= bit;
            } else {
                child_mask &= ~bit;
            }
        }

        void set_filled(uint32_t child_index, bool value) {
            const uint64_t bit = uint64_t{1} << child_index;
            if (value) {
                filled_mask |= bit;
            } else {
                filled_mask &= ~bit;
            }
        }

        void set_child_index(uint32_t child_index, uint32_t index) {
            child_indices[child_index] = index;
        }

        [[nodiscard]] bool has_any_children() const {
            return child_mask != 0;
        }

        [[nodiscard]] bool are_all_filled() const {
            return filled_mask == kLeafFullMask;
        }

        [[nodiscard]] bool are_all_empty() const {
            return filled_mask == 0;
        }
    };

    struct Leaf {
        uint64_t voxel_data = 0;
    };

    [[nodiscard]] uint32_t allocate_node() {
        if (!free_node_indices_.empty()) {
            const uint32_t index = free_node_indices_.back();
            free_node_indices_.pop_back();
            nodes_[index] = Node{};
            return index;
        }

        const uint32_t index = static_cast<uint32_t>(nodes_.size());
        nodes_.push_back(Node{});
        return index;
    }

    [[nodiscard]] uint32_t allocate_leaf() {
        if (!free_leaf_indices_.empty()) {
            const uint32_t index = free_leaf_indices_.back();
            free_leaf_indices_.pop_back();
            leaves_[index] = Leaf{};
            return index;
        }

        const uint32_t index = static_cast<uint32_t>(leaves_.size());
        leaves_.push_back(Leaf{});
        return index;
    }

    void free_node(uint32_t index) {
        if (index == kRootNodeIndex || index >= nodes_.size()) {
            throw std::out_of_range("SVO64: node index out of bounds");
        }
        free_node_indices_.push_back(index);
    }

    void free_leaf(uint32_t index) {
        if (index >= leaves_.size()) {
            throw std::out_of_range("SVO64: leaf index out of bounds");
        }
        free_leaf_indices_.push_back(index);
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
            throw std::out_of_range("SVO64: voxel coordinates out of bounds");
        }
    }

    void validate_max_depth() const {
        if (max_depth_ == 0 || max_depth_ > kMaxSupportedDepth) {
            throw std::out_of_range("SVO64: max_depth out of supported range");
        }
    }

    void free_nodes_to_be_freed(const std::vector<uint32_t>& node_indices,
                                const std::vector<uint32_t>& child_indices,
                                uint32_t leaf_index) {
        const uint64_t leaf_data = leaves_[leaf_index].voxel_data;
        if (leaf_data != 0 && leaf_data != kLeafFullMask) {
            return;
        }

        const bool filled = leaf_data == kLeafFullMask;
        free_leaf(leaf_index);

        for (std::size_t i = node_indices.size(); i-- > 0;) {
            Node& node = nodes_[node_indices[i]];
            const uint32_t child_index = child_indices[i];
            node.set_child_mask(child_index, false);
            node.set_child_index(child_index, 0);
            node.set_filled(child_index, filled);

            if (node.has_any_children()) {
                break;
            }

            if (filled) {
                if (!node.are_all_filled()) {
                    break;
                }
            } else if (!node.are_all_empty()) {
                break;
            }

            if (node_indices[i] == kRootNodeIndex) {
                break;
            }
            free_node(node_indices[i]);
        }
    }

    std::vector<Node> nodes_;
    std::vector<Leaf> leaves_;
    std::vector<uint32_t> free_node_indices_;
    std::vector<uint32_t> free_leaf_indices_;
    uint32_t max_depth_;
};

} // namespace algo::svt
