#pragma once

#include <algo/svt/constants.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace algo::svt {

class HammingSVO {
  public:
    explicit HammingSVO(uint32_t max_depth = DEFAULT_MAX_DEPTH)
        : max_depth_(max_depth) {
        validate_max_depth();
        nodes_.push_back(Node{});
    }

    void set_voxel(uint32_t x, uint32_t y, uint32_t z, bool value) {
        validate_coordinates(x, y, z);

        const uint32_t interior_levels = max_depth_ - 1;
        std::vector<uint32_t> node_indices(interior_levels);
        std::vector<uint32_t> child_indices(interior_levels);

        uint32_t node_index = kRootNodeIndex;
        for (uint32_t depth = 0; depth < interior_levels; ++depth) {
            node_indices[depth] = node_index;
            const uint32_t child_index = get_child_index(x, y, z, depth);
            child_indices[depth] = child_index;

            if (!nodes_[node_index].has_child(child_index)) {
                const uint32_t new_index = (depth + 1 == interior_levels)
                                               ? allocate_leaf_parent()
                                               : allocate_node();
                nodes_[node_index].set_child(child_index, new_index);
            }
            node_index = nodes_[node_index].get_child(child_index);
        }

        const uint32_t leaf_parent_index = node_index;
        LeafParent& parent = leaf_parents_[leaf_parent_index];
        const uint32_t leaf_child_index = get_child_index(x, y, z, interior_levels);

        uint64_t actual_data = 0;
        uint32_t previous_leaf_index = 0;
        bool had_child = parent.has_child(leaf_child_index);

        if (had_child) {
            previous_leaf_index = parent.leaf_indices[leaf_child_index];
            actual_data = reconstruct_actual(leaves_[previous_leaf_index].canonical_data,
                                             parent.error_bits[leaf_child_index]);
        }

        const uint64_t offset = get_leaf_offset(x, y, z);
        const uint64_t bit_mask = uint64_t{1} << offset;
        actual_data = value ? (actual_data | bit_mask)
                             : (actual_data & ~bit_mask);
        actual_data &= kLeafDataMask;

        if (actual_data == 0) {
            if (had_child) {
                release_leaf(previous_leaf_index);
                parent.clear_child(leaf_child_index);
                cleanup_empty_nodes(node_indices, child_indices, leaf_parent_index);
            }
            return;
        }

        uint8_t new_error = 0;
        const uint64_t canonical = canonicalize(actual_data, new_error);
        const uint32_t new_leaf_index = acquire_leaf(canonical);
        if (had_child) {
            release_leaf(previous_leaf_index);
        }
        parent.set_child(leaf_child_index, new_leaf_index, new_error);
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        uint32_t node_index = kRootNodeIndex;
        for (uint32_t depth = 0; depth + 1 < max_depth_; ++depth) {
            const Node& node = nodes_[node_index];
            const uint32_t child_index = get_child_index(x, y, z, depth);
            if (!node.has_child(child_index)) {
                return false;
            }
            node_index = node.get_child(child_index);
        }

        const LeafParent& parent = leaf_parents_[node_index];
        const uint32_t leaf_child_index = get_child_index(x, y, z, max_depth_ - 1);
        if (!parent.has_child(leaf_child_index)) {
            return false;
        }

        uint64_t leaf_data = leaves_[parent.leaf_indices[leaf_child_index]].canonical_data;
        const uint8_t error_bit = parent.error_bits[leaf_child_index];
        if (error_bit != 0) {
            leaf_data ^= uint64_t{1} << (error_bit - 1);
        }

        const uint64_t bit_mask = uint64_t{1} << get_leaf_offset(x, y, z);
        return (leaf_data & bit_mask) != 0;
    }

    [[nodiscard]] std::size_t node_count() const {
        const std::size_t active_nodes = nodes_.size() - free_node_indices_.size();
        const std::size_t active_leaf_parents =
            leaf_parents_.size() - free_leaf_parent_indices_.size();
        return active_nodes + active_leaf_parents;
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
        return LEAF_VOXEL_COUNT << max_depth;
    }

    [[nodiscard]] static constexpr uint32_t
    max_depth_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(
            requested_world_size, LEAF_VOXEL_COUNT, 1, 1,
            kMaxSupportedDepth);
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        std::size_t bytes = 0;
        bytes += nodes_.capacity() * sizeof(Node);
        bytes += free_node_indices_.capacity() * sizeof(uint32_t);
        bytes += leaf_parents_.capacity() * sizeof(LeafParent);
        bytes += free_leaf_parent_indices_.capacity() * sizeof(uint32_t);
        bytes += leaves_.capacity() * sizeof(Leaf);
        bytes += free_leaf_indices_.capacity() * sizeof(uint32_t);
        bytes += leaf_ref_counts_.capacity() * sizeof(uint32_t);
        bytes += canonical_map_.bucket_count() * sizeof(void*);
        bytes += canonical_map_.size() *
                 (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(void*) * 2);
        return bytes;
    }

    [[nodiscard]] std::size_t node_storage_bytes() const {
        return (nodes_.size() - free_node_indices_.size()) * sizeof(Node) +
               (leaf_parents_.size() - free_leaf_parent_indices_.size()) *
                   sizeof(LeafParent) +
               leaf_count() * sizeof(Leaf);
    }

  private:
    static constexpr uint32_t kChildCount = 8;
    static constexpr uint32_t kRootNodeIndex = 0;
    static constexpr uint64_t kLeafDataMask = (uint64_t{1} << 63) - 1;
    static constexpr uint32_t kMaxSupportedDepth = 29;

    struct Node {
        uint32_t child_mask = 0;
        uint32_t child_indices[kChildCount] = {};

        [[nodiscard]] bool has_child(uint32_t child_index) const {
            return (child_mask >> child_index) & 1u;
        }

        [[nodiscard]] uint32_t get_child(uint32_t child_index) const {
            return child_indices[child_index];
        }

        void set_child(uint32_t child_index, uint32_t index) {
            child_indices[child_index] = index;
            child_mask |= (1u << child_index);
        }

        void clear_child(uint32_t child_index) {
            child_mask &= ~(1u << child_index);
            child_indices[child_index] = 0;
        }

        [[nodiscard]] bool has_any_child() const {
            return child_mask != 0;
        }
    };

    struct LeafParent {
        uint32_t leaf_indices[kChildCount] = {};
        uint8_t error_bits[kChildCount] = {};
        uint8_t child_mask = 0;

        void reset() {
            child_mask = 0;
            for (uint32_t i = 0; i < kChildCount; ++i) {
                leaf_indices[i] = 0;
                error_bits[i] = 0;
            }
        }

        [[nodiscard]] bool has_child(uint32_t child_index) const {
            return (child_mask >> child_index) & 1u;
        }

        void set_child(uint32_t child_index, uint32_t leaf_index, uint8_t error_bit) {
            leaf_indices[child_index] = leaf_index;
            error_bits[child_index] = error_bit;
            child_mask |= (1u << child_index);
        }

        void clear_child(uint32_t child_index) {
            child_mask &= ~(1u << child_index);
            leaf_indices[child_index] = 0;
            error_bits[child_index] = 0;
        }

        [[nodiscard]] bool has_any_child() const {
            return child_mask != 0;
        }
    };

    struct Leaf {
        uint64_t canonical_data = 0;
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

    [[nodiscard]] uint32_t allocate_leaf_parent() {
        if (!free_leaf_parent_indices_.empty()) {
            const uint32_t index = free_leaf_parent_indices_.back();
            free_leaf_parent_indices_.pop_back();
            leaf_parents_[index].reset();
            return index;
        }
        const uint32_t index = static_cast<uint32_t>(leaf_parents_.size());
        leaf_parents_.push_back(LeafParent{});
        return index;
    }

    [[nodiscard]] uint32_t acquire_leaf(uint64_t canonical) {
        auto it = canonical_map_.find(canonical);
        if (it != canonical_map_.end()) {
            ++leaf_ref_counts_[it->second];
            return it->second;
        }

        uint32_t index;
        if (!free_leaf_indices_.empty()) {
            index = free_leaf_indices_.back();
            free_leaf_indices_.pop_back();
            leaves_[index].canonical_data = canonical;
            leaf_ref_counts_[index] = 1;
        } else {
            index = static_cast<uint32_t>(leaves_.size());
            leaves_.push_back(Leaf{canonical});
            leaf_ref_counts_.push_back(1);
        }
        canonical_map_[canonical] = index;
        return index;
    }

    void release_leaf(uint32_t index) {
        if (leaf_ref_counts_[index] == 0) {
            return;
        }
        --leaf_ref_counts_[index];
        if (leaf_ref_counts_[index] == 0) {
            canonical_map_.erase(leaves_[index].canonical_data);
            free_leaf_indices_.push_back(index);
        }
    }

    void release_node(uint32_t index) {
        if (index == kRootNodeIndex) {
            return;
        }
        nodes_[index] = Node{};
        free_node_indices_.push_back(index);
    }

    void release_leaf_parent(uint32_t index) {
        leaf_parents_[index].reset();
        free_leaf_parent_indices_.push_back(index);
    }

    void cleanup_empty_nodes(const std::vector<uint32_t>& node_indices,
                              const std::vector<uint32_t>& child_indices,
                              uint32_t leaf_parent_index) {
        if (leaf_parents_[leaf_parent_index].has_any_child()) {
            return;
        }
        release_leaf_parent(leaf_parent_index);

        for (int depth = static_cast<int>(max_depth_) - 2; depth >= 0; --depth) {
            Node& node = nodes_[node_indices[depth]];
            node.clear_child(child_indices[depth]);
            if (node.has_any_child() || node_indices[depth] == kRootNodeIndex) {
                break;
            }
            release_node(node_indices[depth]);
        }
    }

    static uint64_t reconstruct_actual(uint64_t canonical, uint8_t error_bit) {
        if (error_bit == 0) {
            return canonical;
        }
        return canonical ^ (uint64_t{1} << (error_bit - 1));
    }

    [[nodiscard]] static uint64_t canonicalize(uint64_t value, uint8_t& error_bit) {
        value &= kLeafDataMask;
        error_bit = compute_syndrome(value);
        if (error_bit != 0) {
            value ^= uint64_t{1} << (error_bit - 1);
        }
        return value;
    }

    [[nodiscard]] static uint8_t compute_syndrome(uint64_t value) {
        value &= kLeafDataMask;
        uint8_t syndrome = 0;
        for (uint8_t parity_bit = 0; parity_bit < 6; ++parity_bit) {
            const uint32_t mask = 1u << parity_bit;
            bool parity = false;
            for (uint32_t bit_index = 1; bit_index <= 63; ++bit_index) {
                if ((bit_index & mask) == 0) {
                    continue;
                }
                parity ^= bool((value >> (bit_index - 1)) & 1u);
            }
            if (parity) {
                syndrome |= mask;
            }
        }
        return syndrome;
    }

    [[nodiscard]] uint32_t get_child_index(uint32_t x,
                                           uint32_t y,
                                           uint32_t z,
                                           uint32_t depth) const {
        const uint32_t scale = LEAF_VOXEL_COUNT << (max_depth_ - depth - 1);
        const uint32_t x_bit = (x / scale) & 1u;
        const uint32_t y_bit = (y / scale) & 1u;
        const uint32_t z_bit = (z / scale) & 1u;
        return x_bit | (y_bit << 1) | (z_bit << 2);
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
            throw std::out_of_range("HammingSVO: voxel coordinates out of bounds");
        }
    }

    void validate_max_depth() const {
        if (max_depth_ == 0 || max_depth_ > kMaxSupportedDepth) {
            throw std::out_of_range("HammingSVO: max_depth out of supported range");
        }
    }

    std::vector<Node> nodes_;
    std::vector<uint32_t> free_node_indices_;
    std::vector<LeafParent> leaf_parents_;
    std::vector<uint32_t> free_leaf_parent_indices_;
    std::vector<Leaf> leaves_;
    std::vector<uint32_t> free_leaf_indices_;
    std::vector<uint32_t> leaf_ref_counts_;
    std::unordered_map<uint64_t, uint32_t> canonical_map_;
    uint32_t max_depth_;
};

} // namespace algo::svt
