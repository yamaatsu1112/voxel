#pragma once

#include <algo/svt/constants.hpp>
#include <algo/svt/svdag/detail/build_types.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace algo::svt {

class SVDAGBuilder;

namespace detail {
class LabeledSVDAGBuildPolicy;
}

class LabeledSVDAG {
  public:
    explicit LabeledSVDAG(uint32_t max_level = DEFAULT_MAX_DEPTH)
        : max_level_(max_level) {
        validate_max_level();
        node_counts_by_level_.resize(max_level_);
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        uint32_t ptr = root_;
        while (true) {
            if (ptr == kEmpty) {
                return false;
            }
            if (ptr == kFilled) {
                return true;
            }

            const uint32_t level = level_at(dag_words_, ptr);
            if (level >= max_level_) {
                throw std::logic_error("LabeledSVDAG: node level is invalid");
            }

            ptr = child_at(dag_words_, ptr, get_child_index(x, y, z, level));
        }
    }

    [[nodiscard]] std::size_t node_count() const { return node_count_; }

    [[nodiscard]] std::vector<std::size_t> node_counts_by_level() const {
        return node_counts_by_level_;
    }

    [[nodiscard]] std::size_t leaf_count() const { return 0; }

    [[nodiscard]] std::size_t node_storage_bytes() const {
        return dag_words_.size() * sizeof(uint32_t);
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        return dag_words_.capacity() * sizeof(uint32_t) +
               node_counts_by_level_.capacity() * sizeof(std::size_t);
    }

    [[nodiscard]] uint32_t max_level() const { return max_level_; }

    [[nodiscard]] uint32_t max_depth() const { return max_level_; }

    [[nodiscard]] uint32_t world_voxel_count() const {
        return world_voxel_count(max_level_);
    }

    [[nodiscard]] static constexpr uint32_t
    world_voxel_count(uint32_t max_level) {
        return max_level < 32 ? (uint32_t{1} << max_level) : 0;
    }

    [[nodiscard]] static constexpr uint32_t
    max_level_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(requested_world_size, 1, 1, 0,
                                            kMaxSupportedLevel);
    }

    [[nodiscard]] static constexpr uint32_t
    max_depth_for_world_size(uint32_t requested_world_size) {
        return max_level_for_world_size(requested_world_size);
    }

  private:
    static constexpr uint32_t kChildCount = 8;
    static constexpr uint32_t kEmpty = 0;
    static constexpr uint32_t kFilled = 1;
    static constexpr uint32_t kFirstNode = 2;
    static constexpr uint32_t kMaxSupportedLevel = 31;
    static constexpr uint32_t kChildMaskShift = 0;
    static constexpr uint32_t kFilledMaskShift = 8;
    static constexpr uint32_t kLevelShift = 16;
    static constexpr uint32_t kMaskMask = 0xFFu;
    static constexpr uint32_t kLevelMask = 0x1Fu;
    static constexpr uint32_t kHeaderReservedMask = 0xFFE00000u;

    friend class SVDAGBuilder;
    friend class detail::LabeledSVDAGBuildPolicy;

    LabeledSVDAG(uint32_t max_level, uint32_t root,
                 std::vector<uint32_t> dag_words,
                 std::vector<std::size_t> node_counts_by_level,
                 std::size_t node_count)
        : dag_words_(std::move(dag_words)),
          node_counts_by_level_(std::move(node_counts_by_level)),
          node_count_(node_count), root_(root), max_level_(max_level) {
        validate_max_level();
    }

    [[nodiscard]] static uint32_t checked_word_offset(std::size_t offset) {
        if (offset > static_cast<std::size_t>(UINT32_MAX - kFirstNode)) {
            throw std::overflow_error("LabeledSVDAG: node offset overflow");
        }
        return static_cast<uint32_t>(offset);
    }

    [[nodiscard]] static uint32_t mask_bit(uint32_t child_index) {
        return uint32_t{1} << child_index;
    }

    [[nodiscard]] static uint32_t child_mask(uint32_t header) {
        return (header >> kChildMaskShift) & kMaskMask;
    }

    [[nodiscard]] static uint32_t filled_mask(uint32_t header) {
        return (header >> kFilledMaskShift) & kMaskMask;
    }

    [[nodiscard]] static uint32_t node_level(uint32_t header) {
        return (header >> kLevelShift) & kLevelMask;
    }

    [[nodiscard]] static uint32_t
    header_at(const std::vector<uint32_t>& words, uint32_t ptr) {
        if (ptr < kFirstNode) {
            throw std::out_of_range("LabeledSVDAG: node pointer out of bounds");
        }

        const uint32_t offset = ptr - kFirstNode;
        if (offset >= words.size()) {
            throw std::out_of_range("LabeledSVDAG: node pointer out of bounds");
        }

        const uint32_t header = words[offset];
        if ((header & kHeaderReservedMask) != 0) {
            throw std::logic_error(
                "LabeledSVDAG: packed node header has reserved bits set");
        }

        const uint32_t children = child_mask(header);
        const uint32_t filled = filled_mask(header);
        if ((filled & ~children) != 0) {
            throw std::logic_error(
                "LabeledSVDAG: packed node filled mask is invalid");
        }

        return header;
    }

    [[nodiscard]] static uint32_t level_at(const std::vector<uint32_t>& words,
                                           uint32_t ptr) {
        return node_level(header_at(words, ptr));
    }

    [[nodiscard]] static uint32_t child_at(const std::vector<uint32_t>& words,
                                           uint32_t ptr,
                                           uint32_t child_index) {
        const uint32_t offset = ptr - kFirstNode;
        const uint32_t header = header_at(words, ptr);
        const uint32_t children = child_mask(header);
        const uint32_t filled = filled_mask(header);

        const uint32_t bit = mask_bit(child_index);
        if ((children & bit) == 0) {
            return kEmpty;
        }
        if ((filled & bit) != 0) {
            return kFilled;
        }

        const uint32_t pointer_mask = children & ~filled;
        const uint32_t rank = std::popcount(pointer_mask & (bit - 1u));
        const std::size_t child_offset =
            static_cast<std::size_t>(offset) + 1u + rank;
        if (child_offset >= words.size()) {
            throw std::out_of_range(
                "LabeledSVDAG: packed child pointer out of bounds");
        }
        return words[child_offset];
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
        const uint32_t voxel_count = LabeledSVDAG::world_voxel_count(max_level_);
        if (x >= voxel_count || y >= voxel_count || z >= voxel_count) {
            throw std::out_of_range(
                "LabeledSVDAG: voxel coordinates out of bounds");
        }
    }

    void validate_max_level() const {
        if (max_level_ > kMaxSupportedLevel) {
            throw std::out_of_range(
                "LabeledSVDAG: max_level out of supported range");
        }
    }

    std::vector<uint32_t> dag_words_;
    std::vector<std::size_t> node_counts_by_level_;
    std::size_t node_count_ = 0;
    uint32_t root_ = kEmpty;
    uint32_t max_level_ = 0;
};

namespace detail {

class LabeledSVDAGBuildPolicy {
  public:
    using Result = LabeledSVDAG;

    explicit LabeledSVDAGBuildPolicy(uint32_t max_level)
        : node_counts_by_level_(max_level) {}

    [[nodiscard]] static Result finish_empty(uint32_t max_level,
                                             uint32_t root) {
        return LabeledSVDAG(max_level, root, {}, {}, 0);
    }

    [[nodiscard]] bool reduce(const SVDAGBuildNode& key,
                              uint32_t& result) const {
        if (!has_uniform_children(key)) {
            return false;
        }
        result = key.child[0];
        return true;
    }

    [[nodiscard]] uint32_t append(uint32_t level,
                                  const SVDAGBuildNode& node) {
        if (level >= node_counts_by_level_.size()) {
            throw std::out_of_range("LabeledSVDAG: DAG level out of bounds");
        }

        const uint32_t offset =
            LabeledSVDAG::checked_word_offset(dag_words_.size());

        uint32_t child_mask = 0;
        uint32_t filled_mask = 0;
        std::array<uint32_t, LabeledSVDAG::kChildCount> child_ptrs = {};
        uint32_t child_ptr_count = 0;

        for (uint32_t child_index = 0;
             child_index < LabeledSVDAG::kChildCount; ++child_index) {
            const uint32_t child = node.child[child_index];
            if (child == LabeledSVDAG::kEmpty) {
                continue;
            }

            const uint32_t bit = LabeledSVDAG::mask_bit(child_index);
            child_mask |= bit;
            if (child == LabeledSVDAG::kFilled) {
                filled_mask |= bit;
                continue;
            }
            if (child < LabeledSVDAG::kFirstNode) {
                throw std::logic_error("LabeledSVDAG: invalid child pointer");
            }
            child_ptrs[child_ptr_count++] = child;
        }

        const uint32_t header =
            (child_mask << LabeledSVDAG::kChildMaskShift) |
            (filled_mask << LabeledSVDAG::kFilledMaskShift) |
            (level << LabeledSVDAG::kLevelShift);
        dag_words_.push_back(header);
        dag_words_.insert(dag_words_.end(), child_ptrs.begin(),
                          child_ptrs.begin() + child_ptr_count);
        ++node_counts_by_level_[level];
        ++node_count_;

        return LabeledSVDAG::kFirstNode + offset;
    }

    [[nodiscard]] Result finish(uint32_t max_level, uint32_t root) {
        return LabeledSVDAG(max_level, root, std::move(dag_words_),
                            std::move(node_counts_by_level_), node_count_);
    }

  private:
    std::vector<uint32_t> dag_words_;
    std::vector<std::size_t> node_counts_by_level_;
    std::size_t node_count_ = 0;
};

} // namespace detail

} // namespace algo::svt
