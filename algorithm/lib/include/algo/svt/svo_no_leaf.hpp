#pragma once

#include <algo/svt/constants.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace algo::svt {

constexpr uint32_t NO_LEAF_CHILD_INDEX_OFFSET = 0;
constexpr uint32_t NO_LEAF_CHILD_MASK_OFFSET = 30;
constexpr uint32_t NO_LEAF_FILLED_OFFSET = 31;
constexpr uint32_t NO_LEAF_CHILD_INDEX_MASK = 0x3FFFFFFFu;
constexpr uint32_t NO_LEAF_CHILD_MASK_BIT = 1u << NO_LEAF_CHILD_MASK_OFFSET;
constexpr uint32_t NO_LEAF_FILLED_BIT = 1u << NO_LEAF_FILLED_OFFSET;

class SVONoLeaf {
  public:
    explicit SVONoLeaf(uint32_t max_depth = DEFAULT_MAX_DEPTH) : max_depth_(max_depth) {
        validate_max_depth();
        nodes_.push_back(Node{});
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

            if (nodes_[node_index].has_child(child_index)) {
                node_index = nodes_[node_index].get_child_index(child_index);
                node_indices[depth + 1] = node_index;
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
            new_node.set_uniform(filled);
            node_index = new_node_index;
            node_indices[depth + 1] = node_index;
        }

        Node& occupancy = nodes_[node_index];
        const uint32_t bit_offset = get_terminal_offset(x, y, z);
        if (occupancy.is_filled(bit_offset) == value) {
            return;
        }
        occupancy.set_filled(bit_offset, value);

        collapse_path(node_indices, child_indices);
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        uint32_t node_index = kRootNodeIndex;

        for (uint32_t depth = 0; depth < max_depth_; ++depth) {
            const Node& node = nodes_[node_index];
            const uint32_t child_index = get_child_index(x, y, z, depth);
            if (!node.has_child(child_index)) {
                return node.is_filled(child_index);
            }
            node_index = node.get_child_index(child_index);
        }

        return nodes_[node_index].is_filled(get_terminal_offset(x, y, z));
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

    [[nodiscard]] static constexpr uint32_t world_voxel_count(uint32_t max_depth) {
        return TERMINAL_VOXEL_COUNT << max_depth;
    }

    [[nodiscard]] static constexpr uint32_t
    max_depth_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(
            requested_world_size, TERMINAL_VOXEL_COUNT, 1, 1,
            kMaxSupportedDepth);
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        return nodes_.capacity() * sizeof(Node) +
               free_node_indices_.capacity() * sizeof(uint32_t);
    }

    [[nodiscard]] std::size_t node_storage_bytes() const {
        return node_count() * sizeof(Node);
    }

  private:
    static constexpr uint32_t kChildCount = 8;
    static constexpr uint32_t kRootNodeIndex = 0;
    static constexpr uint32_t kMaxSupportedDepth = 29;
    static constexpr uint32_t TERMINAL_VOXEL_COUNT = 2;

    struct ChildInfo {
        uint32_t info = 0;

        [[nodiscard]] bool has_child() const {
            return (info & NO_LEAF_CHILD_MASK_BIT) != 0;
        }

        [[nodiscard]] bool is_filled() const {
            return (info & NO_LEAF_FILLED_BIT) != 0;
        }

        [[nodiscard]] uint32_t get_child_index() const {
            return (info & NO_LEAF_CHILD_INDEX_MASK) >> NO_LEAF_CHILD_INDEX_OFFSET;
        }

        void set_child_mask(bool value) {
            if (value) {
                info |= NO_LEAF_CHILD_MASK_BIT;
            } else {
                info &= ~NO_LEAF_CHILD_MASK_BIT;
            }
        }

        void set_filled(bool value) {
            if (value) {
                info |= NO_LEAF_FILLED_BIT;
            } else {
                info &= ~NO_LEAF_FILLED_BIT;
            }
        }

        void set_child_index(uint32_t index) {
            info = (info & ~NO_LEAF_CHILD_INDEX_MASK) |
                   ((index << NO_LEAF_CHILD_INDEX_OFFSET) &
                    NO_LEAF_CHILD_INDEX_MASK);
        }
    };

    struct Node {
        ChildInfo child_info[kChildCount];

        [[nodiscard]] bool has_child(uint32_t child_index) const {
            return child_info[child_index].has_child();
        }

        [[nodiscard]] bool is_filled(uint32_t child_index) const {
            return child_info[child_index].is_filled();
        }

        [[nodiscard]] uint32_t get_child_index(uint32_t child_index) const {
            return child_info[child_index].get_child_index();
        }

        void set_child_mask(uint32_t child_index, bool value) {
            child_info[child_index].set_child_mask(value);
        }

        void set_filled(uint32_t child_index, bool value) {
            child_info[child_index].set_filled(value);
        }

        void set_child_index(uint32_t child_index, uint32_t index) {
            child_info[child_index].set_child_index(index);
        }

        void set_uniform(bool filled) {
            for (uint32_t i = 0; i < kChildCount; ++i) {
                set_child_mask(i, false);
                set_child_index(i, 0);
                set_filled(i, filled);
            }
        }

        [[nodiscard]] bool has_any_children() const {
            for (uint32_t i = 0; i < kChildCount; ++i) {
                if (has_child(i)) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool are_all_filled() const {
            for (uint32_t i = 0; i < kChildCount; ++i) {
                if (!is_filled(i)) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] bool are_all_empty() const {
            for (uint32_t i = 0; i < kChildCount; ++i) {
                if (is_filled(i)) {
                    return false;
                }
            }
            return true;
        }
    };

    [[nodiscard]] uint32_t allocate_node() {
        if (!free_node_indices_.empty()) {
            const uint32_t index = free_node_indices_.back();
            free_node_indices_.pop_back();
            nodes_[index] = Node{};
            return index;
        }

        const auto index = static_cast<uint32_t>(nodes_.size());
        if (index > NO_LEAF_CHILD_INDEX_MASK) {
            throw std::overflow_error("SVONoLeaf: node index overflow");
        }
        nodes_.push_back(Node{});
        return index;
    }

    void free_node(uint32_t index) {
        if (index == kRootNodeIndex || index >= nodes_.size()) {
            throw std::out_of_range("SVONoLeaf: node index out of bounds");
        }
        free_node_indices_.push_back(index);
    }

    [[nodiscard]] uint32_t get_child_index(uint32_t x,
                                           uint32_t y,
                                           uint32_t z,
                                           uint32_t depth) const {
        const uint32_t scale =
            TERMINAL_VOXEL_COUNT << (max_depth_ - depth - 1);
        const uint32_t x_bit = (x / scale) & 1u;
        const uint32_t y_bit = (y / scale) & 1u;
        const uint32_t z_bit = (z / scale) & 1u;
        return x_bit | (y_bit << 1) | (z_bit << 2);
    }

    [[nodiscard]] static uint32_t get_terminal_offset(uint32_t x,
                                                      uint32_t y,
                                                      uint32_t z) {
        const uint32_t x_offset = x & 1u;
        const uint32_t y_offset = y & 1u;
        const uint32_t z_offset = z & 1u;
        return x_offset | (y_offset << 1) | (z_offset << 2);
    }

    void validate_coordinates(uint32_t x, uint32_t y, uint32_t z) const {
        if (x >= world_voxel_count() || y >= world_voxel_count() ||
            z >= world_voxel_count()) {
            throw std::out_of_range("SVONoLeaf: voxel coordinates out of bounds");
        }
    }

    void validate_max_depth() const {
        if (max_depth_ == 0 || max_depth_ > kMaxSupportedDepth) {
            throw std::out_of_range("SVONoLeaf: max_depth out of supported range");
        }
    }

    void collapse_path(const std::vector<uint32_t>& node_indices,
                       const std::vector<uint32_t>& child_indices) {
        for (std::size_t i = node_indices.size(); i-- > 1;) {
            Node& node = nodes_[node_indices[i]];
            if (node.has_any_children()) {
                break;
            }

            if (!node.are_all_empty() && !node.are_all_filled()) {
                break;
            }

            const bool filled = node.are_all_filled();
            Node& parent = nodes_[node_indices[i - 1]];
            const uint32_t child_index = child_indices[i - 1];
            parent.set_child_mask(child_index, false);
            parent.set_child_index(child_index, 0);
            parent.set_filled(child_index, filled);
            free_node(node_indices[i]);
        }
    }

    std::vector<Node> nodes_;
    std::vector<uint32_t> free_node_indices_;
    uint32_t max_depth_;
};

} // namespace algo::svt
