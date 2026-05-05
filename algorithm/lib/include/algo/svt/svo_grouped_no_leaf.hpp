#pragma once

#include <algo/svt/constants.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace algo::svt {

class SVOGroupedNoLeaf {
  public:
    explicit SVOGroupedNoLeaf(uint32_t max_depth = DEFAULT_MAX_DEPTH)
        : max_depth_(max_depth) {
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

            if (!nodes_[node_index].has_child(child_index)) {
                if (nodes_[node_index].is_filled(child_index) == value) {
                    return;
                }
                if (nodes_[node_index].head_pointer == 0) {
                    static_cast<void>(allocate_node_block(node_index));
                }
                nodes_[node_index].set_child_mask(child_index, true);
            }

            node_index = nodes_[node_index].get_child_node_index(child_index);
            node_indices[depth + 1] = node_index;
        }

        Node& terminal = nodes_[node_index];
        const uint32_t bit_offset = get_terminal_offset(x, y, z);
        if (terminal.is_filled(bit_offset) == value) {
            return;
        }
        terminal.set_filled(bit_offset, value);

        collapse_uniform_path(node_indices, child_indices);
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
            node_index = node.get_child_node_index(child_index);
        }

        return nodes_[node_index].is_filled(get_terminal_offset(x, y, z));
    }

    [[nodiscard]] std::size_t node_count() const {
        return nodes_.size() - free_node_block_indices_.size() * kChildCount;
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
        return kTerminalVoxelCount << max_depth;
    }

    [[nodiscard]] static constexpr uint32_t
    max_depth_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(
            requested_world_size, kTerminalVoxelCount, 1, 1,
            kMaxSupportedDepth);
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        return nodes_.capacity() * sizeof(Node) +
               free_node_block_indices_.capacity() * sizeof(uint32_t);
    }

    [[nodiscard]] std::size_t node_storage_bytes() const {
        return node_count() * sizeof(Node);
    }

  private:
    static constexpr uint32_t kChildCount = 8;
    static constexpr uint32_t kRootNodeIndex = 0;
    static constexpr uint32_t kFilledMaskOffset = 8;
    static constexpr uint32_t kFilledMaskBits = 0xFF00u;
    static constexpr uint32_t kMaxSupportedDepth = 29;
    static constexpr uint32_t kTerminalVoxelCount = 2;

    struct Node {
        uint32_t head_pointer = 0;
        uint32_t masks = 0;

        [[nodiscard]] bool has_child(uint32_t child_index) const {
            return (masks & (uint32_t{1} << child_index)) != 0;
        }

        [[nodiscard]] bool is_filled(uint32_t child_index) const {
            return (masks & (uint32_t{1}
                             << (child_index + kFilledMaskOffset))) != 0;
        }

        [[nodiscard]] uint32_t get_child_node_index(uint32_t child_index) const {
            return head_pointer + child_index;
        }

        void set_child_mask(uint32_t child_index, bool value) {
            const uint32_t bit = uint32_t{1} << child_index;
            if (value) {
                masks |= bit;
            } else {
                masks &= ~bit;
            }
        }

        void set_filled(uint32_t child_index, bool value) {
            const uint32_t bit =
                uint32_t{1} << (child_index + kFilledMaskOffset);
            if (value) {
                masks |= bit;
            } else {
                masks &= ~bit;
            }
        }

        void set_uniform(bool filled) {
            head_pointer = 0;
            masks = filled ? kFilledMaskBits : 0;
        }

        [[nodiscard]] bool has_any_children() const {
            return (masks & 0xFFu) != 0;
        }

        [[nodiscard]] bool are_all_empty() const {
            return (masks & kFilledMaskBits) == 0;
        }

        [[nodiscard]] bool are_all_filled() const {
            return (masks & kFilledMaskBits) == kFilledMaskBits;
        }
    };

    [[nodiscard]] uint32_t allocate_node_block(uint32_t parent_index) {
        uint32_t block_index = 0;
        if (!free_node_block_indices_.empty()) {
            block_index = free_node_block_indices_.back();
            free_node_block_indices_.pop_back();
        } else {
            block_index = static_cast<uint32_t>(nodes_.size());
            if (block_index == 0 || block_index > UINT32_MAX - kChildCount) {
                throw std::overflow_error(
                    "SVOGroupedNoLeaf: node block index overflow");
            }
            nodes_.resize(nodes_.size() + kChildCount);
        }

        if (block_index == 0 || block_index + kChildCount > nodes_.size()) {
            throw std::out_of_range(
                "SVOGroupedNoLeaf: node block index out of bounds");
        }

        const Node parent = nodes_[parent_index];
        for (uint32_t child_index = 0; child_index < kChildCount; ++child_index) {
            Node& child = nodes_[block_index + child_index];
            child = Node{};
            child.set_uniform(parent.is_filled(child_index));
        }
        nodes_[parent_index].head_pointer = block_index;
        return block_index;
    }

    void free_node_block(uint32_t block_index) {
        if (block_index == 0 || block_index + kChildCount > nodes_.size()) {
            throw std::out_of_range(
                "SVOGroupedNoLeaf: node block index out of bounds");
        }
        free_node_block_indices_.push_back(block_index);
    }

    [[nodiscard]] uint32_t get_child_index(uint32_t x,
                                           uint32_t y,
                                           uint32_t z,
                                           uint32_t depth) const {
        const uint32_t scale =
            kTerminalVoxelCount << (max_depth_ - depth - 1);
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
            throw std::out_of_range(
                "SVOGroupedNoLeaf: voxel coordinates out of bounds");
        }
    }

    void validate_max_depth() const {
        if (max_depth_ == 0 || max_depth_ > kMaxSupportedDepth) {
            throw std::out_of_range(
                "SVOGroupedNoLeaf: max_depth out of supported range");
        }
    }

    void collapse_uniform_path(const std::vector<uint32_t>& node_indices,
                               const std::vector<uint32_t>& child_indices) {
        for (std::size_t depth = max_depth_;; --depth) {
            const uint32_t current_index = node_indices[depth];
            const Node current = nodes_[current_index];

            if (current.has_any_children()) {
                break;
            }

            bool current_full = false;
            if (current.are_all_filled()) {
                current_full = true;
            } else if (!current.are_all_empty()) {
                break;
            }

            const uint32_t child_block_index = current.head_pointer;
            if (child_block_index != 0) {
                free_node_block(child_block_index);
                nodes_[current_index].set_uniform(current_full);
            }

            if (depth == 0) {
                break;
            }

            const uint32_t parent_index = node_indices[depth - 1];
            const uint32_t child_index = child_indices[depth - 1];
            nodes_[parent_index].set_child_mask(child_index, false);
            nodes_[parent_index].set_filled(child_index, current_full);
        }
    }

    std::vector<Node> nodes_;
    std::vector<uint32_t> free_node_block_indices_;
    uint32_t max_depth_;
};

} // namespace algo::svt
