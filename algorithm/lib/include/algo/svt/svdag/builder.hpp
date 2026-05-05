#pragma once

#include <algo/svt/svdag/classic.hpp>
#include <algo/svt/svdag/labeled.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace algo::svt {

class SVDAGBuilder {
  public:
    explicit SVDAGBuilder(uint32_t max_level = DEFAULT_MAX_DEPTH)
        : max_level_(max_level) {
        validate_max_level();
        if (max_level_ == 0) {
            return;
        }

        build_levels_.resize(max_level_);
        build_levels_[0].push_back(Node{});
        root_ = kFirstNode;
    }

    void set_voxel(uint32_t x, uint32_t y, uint32_t z, bool value) {
        validate_coordinates(x, y, z);

        const uint32_t target = value ? kFilled : kEmpty;
        if (max_level_ == 0) {
            root_ = target;
            return;
        }

        uint32_t node_ptr = root_;
        for (uint32_t level = 0; level < max_level_; ++level) {
            Node& node = mutable_node_at(build_levels_, level, node_ptr);
            const uint32_t child_index = get_child_index(x, y, z, level);
            uint32_t& child = node.child[child_index];

            if (level + 1 == max_level_) {
                child = target;
                return;
            }

            if (is_uniform_ptr(child)) {
                if (child == target) {
                    return;
                }

                child = append_build_node(level + 1, child);
            }

            node_ptr = child;
        }
    }

    [[nodiscard]] SVDAG build() const {
        return build_with_policy<detail::ClassicSVDAGBuildPolicy>();
    }

    [[nodiscard]] LabeledSVDAG build_labeled() const {
        return build_with_policy<detail::LabeledSVDAGBuildPolicy>();
    }

    [[nodiscard]] static constexpr uint32_t
    max_level_for_world_size(uint32_t requested_world_size) {
        return SVDAG::max_level_for_world_size(requested_world_size);
    }

  private:
    using Node = detail::SVDAGBuildNode;
    using SortEntry = detail::SVDAGBuildSortEntry;

    static_assert(sizeof(Node) == sizeof(uint32_t) * SVDAG::kChildCount);

    static constexpr uint32_t kEmpty = SVDAG::kEmpty;
    static constexpr uint32_t kFilled = SVDAG::kFilled;
    static constexpr uint32_t kFirstNode = SVDAG::kFirstNode;
    static constexpr uint32_t kMaxSupportedLevel = SVDAG::kMaxSupportedLevel;

    [[nodiscard]] static bool is_uniform_ptr(uint32_t ptr) {
        return SVDAG::is_uniform_ptr(ptr);
    }

    template <typename Policy>
    [[nodiscard]] typename Policy::Result build_with_policy() const {
        if (max_level_ == 0) {
            return Policy::finish_empty(max_level_, root_);
        }

        Policy policy(max_level_);
        std::vector<std::vector<uint32_t>> remap(max_level_);
        for (uint32_t reverse_level = max_level_; reverse_level > 0;
             --reverse_level) {
            const uint32_t level = reverse_level - 1;
            remap[level].assign(build_levels_[level].size(), kEmpty);

            std::vector<SortEntry> entries;
            entries.reserve(build_levels_[level].size());

            for (uint32_t old_index = 0;
                 old_index < build_levels_[level].size(); ++old_index) {
                Node key = build_levels_[level][old_index];
                if (level + 1 < max_level_) {
                    canonicalize_child_ptrs(key, remap[level + 1]);
                }

                uint32_t reduced = kEmpty;
                if (policy.reduce(key, reduced)) {
                    remap[level][old_index] = reduced;
                    continue;
                }

                entries.push_back(SortEntry{key, old_index});
            }

            std::sort(entries.begin(), entries.end(),
                      [](const SortEntry& lhs, const SortEntry& rhs) {
                          return lhs.key.child < rhs.key.child;
                      });

            for (std::size_t i = 0; i < entries.size();) {
                const Node key = entries[i].key;
                const uint32_t new_ptr = policy.append(level, key);
                do {
                    remap[level][entries[i].old_index] = new_ptr;
                    ++i;
                } while (i < entries.size() &&
                         entries[i].key.child == key.child);
            }
        }

        return policy.finish(max_level_, remap[0][0]);
    }

    [[nodiscard]] static uint32_t checked_level_size(std::size_t size) {
        if (size > static_cast<std::size_t>(UINT32_MAX - kFirstNode)) {
            throw std::overflow_error("SVDAG: node index overflow");
        }
        return static_cast<uint32_t>(size);
    }

    static void
    canonicalize_child_ptrs(Node& node,
                            const std::vector<uint32_t>& child_remap) {
        for (uint32_t& child : node.child) {
            if (child < kFirstNode) {
                continue;
            }

            const uint32_t old_child_index = child - kFirstNode;
            if (old_child_index >= child_remap.size()) {
                throw std::logic_error("SVDAG: missing child remap entry");
            }
            child = child_remap[old_child_index];
        }
    }

    [[nodiscard]] uint32_t append_build_node(uint32_t level,
                                             uint32_t fill_value) {
        if (level >= build_levels_.size()) {
            throw std::out_of_range("SVDAG: build level out of bounds");
        }
        const uint32_t index = checked_level_size(build_levels_[level].size());
        Node node;
        node.child.fill(fill_value);
        build_levels_[level].push_back(node);
        return kFirstNode + index;
    }

    [[nodiscard]] static Node&
    mutable_node_at(std::vector<std::vector<Node>>& levels, uint32_t level,
                    uint32_t ptr) {
        if (ptr < kFirstNode || level >= levels.size()) {
            throw std::out_of_range("SVDAG: node pointer out of bounds");
        }

        const uint32_t index = ptr - kFirstNode;
        if (index >= levels[level].size()) {
            throw std::out_of_range("SVDAG: node pointer out of bounds");
        }
        return levels[level][index];
    }

    [[nodiscard]] uint32_t get_child_index(uint32_t x, uint32_t y, uint32_t z,
                                           uint32_t level) const {
        const uint32_t bit = max_level_ - level - 1;
        const uint32_t x_bit = (x >> bit) & 1u;
        const uint32_t y_bit = (y >> bit) & 1u;
        const uint32_t z_bit = (z >> bit) & 1u;
        return x_bit | (y_bit << 1) | (z_bit << 2);
    }

    void validate_coordinates(uint32_t x, uint32_t y, uint32_t z) const {
        const uint32_t voxel_count = SVDAG::world_voxel_count(max_level_);
        if (x >= voxel_count || y >= voxel_count || z >= voxel_count) {
            throw std::out_of_range("SVDAG: voxel coordinates out of bounds");
        }
    }

    void validate_max_level() const {
        if (max_level_ > kMaxSupportedLevel) {
            throw std::out_of_range("SVDAG: max_level out of supported range");
        }
    }

    std::vector<std::vector<Node>> build_levels_;
    uint32_t root_ = kEmpty;
    uint32_t max_level_ = 0;
};

} // namespace algo::svt
