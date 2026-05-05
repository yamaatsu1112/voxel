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
class ClassicSVDAGBuildPolicy;
}

class SVDAG {
  public:
    explicit SVDAG(uint32_t max_level = DEFAULT_MAX_DEPTH)
        : max_level_(max_level) {
        validate_max_level();
        dag_levels_.resize(max_level_);
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        uint32_t ptr = root_;
        for (uint32_t level = 0; level < max_level_; ++level) {
            if (ptr == kEmpty) {
                return false;
            }
            if (ptr == kFilled) {
                return true;
            }

            ptr = child_at(dag_levels_, level, ptr,
                           get_child_index(x, y, z, level));
        }

        return ptr == kFilled;
    }

    [[nodiscard]] std::size_t node_count() const {
        return count_nodes(dag_levels_);
    }

    [[nodiscard]] std::vector<std::size_t> node_counts_by_level() const {
        std::vector<std::size_t> counts;
        counts.reserve(dag_levels_.size());
        for (const auto& level : dag_levels_) {
            counts.push_back(level.node_count);
        }
        return counts;
    }

    [[nodiscard]] std::size_t leaf_count() const { return 0; }

    [[nodiscard]] std::size_t node_storage_bytes() const {
        return packed_node_storage_bytes(dag_levels_);
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        return level_storage_bytes(dag_levels_);
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
    struct PackedLevel {
        std::vector<uint32_t> words;
        std::size_t node_count = 0;
    };

    static constexpr uint32_t kChildCount = 8;
    static constexpr uint32_t kEmpty = 0;
    static constexpr uint32_t kFilled = 1;
    static constexpr uint32_t kFirstNode = 2;
    static constexpr uint32_t kMaxSupportedLevel = 31;
    static constexpr uint32_t kChildMaskShift = 0;
    static constexpr uint32_t kFilledMaskShift = 8;
    static constexpr uint32_t kMaskMask = 0xFFu;
    static constexpr uint32_t kHeaderReservedMask = 0xFFFF0000u;

    friend class SVDAGBuilder;
    friend class detail::ClassicSVDAGBuildPolicy;

    SVDAG(uint32_t max_level, uint32_t root,
          std::vector<PackedLevel> dag_levels)
        : dag_levels_(std::move(dag_levels)), root_(root),
          max_level_(max_level) {
        validate_max_level();
    }

    [[nodiscard]] static bool is_uniform_ptr(uint32_t ptr) {
        return ptr == kEmpty || ptr == kFilled;
    }

    [[nodiscard]] static uint32_t checked_word_offset(std::size_t offset) {
        if (offset > static_cast<std::size_t>(UINT32_MAX - kFirstNode)) {
            throw std::overflow_error("SVDAG: node offset overflow");
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

    [[nodiscard]] static uint32_t
    child_at(const std::vector<PackedLevel>& levels, uint32_t level,
             uint32_t ptr, uint32_t child_index) {
        if (ptr < kFirstNode || level >= levels.size()) {
            throw std::out_of_range("SVDAG: node pointer out of bounds");
        }

        const uint32_t offset = ptr - kFirstNode;
        const PackedLevel& packed_level = levels[level];
        if (offset >= packed_level.words.size()) {
            throw std::out_of_range("SVDAG: node pointer out of bounds");
        }

        const uint32_t header = packed_level.words[offset];
        if ((header & kHeaderReservedMask) != 0) {
            throw std::logic_error(
                "SVDAG: packed node header has reserved bits set");
        }

        const uint32_t children = child_mask(header);
        const uint32_t filled = filled_mask(header);
        if ((filled & ~children) != 0) {
            throw std::logic_error("SVDAG: packed node filled mask is invalid");
        }

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
        if (child_offset >= packed_level.words.size()) {
            throw std::out_of_range(
                "SVDAG: packed child pointer out of bounds");
        }
        return packed_level.words[child_offset];
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

    [[nodiscard]] static std::size_t
    count_nodes(const std::vector<PackedLevel>& levels) {
        std::size_t count = 0;
        for (const auto& level : levels) {
            count += level.node_count;
        }
        return count;
    }

    [[nodiscard]] static std::size_t
    packed_node_storage_bytes(const std::vector<PackedLevel>& levels) {
        std::size_t bytes = 0;
        for (const auto& level : levels) {
            bytes += level.words.size() * sizeof(uint32_t);
        }
        return bytes;
    }

    [[nodiscard]] static std::size_t
    level_storage_bytes(const std::vector<PackedLevel>& levels) {
        std::size_t bytes = 0;
        for (const auto& level : levels) {
            bytes += level.words.capacity() * sizeof(uint32_t);
        }
        return bytes;
    }

    std::vector<PackedLevel> dag_levels_;
    uint32_t root_ = kEmpty;
    uint32_t max_level_ = 0;
};

namespace detail {

class ClassicSVDAGBuildPolicy {
  public:
    using Result = SVDAG;

    explicit ClassicSVDAGBuildPolicy(uint32_t max_level)
        : dag_levels_(max_level) {}

    [[nodiscard]] static Result finish_empty(uint32_t max_level,
                                             uint32_t root) {
        return SVDAG(max_level, root, {});
    }

    [[nodiscard]] bool reduce(const SVDAGBuildNode& key,
                              uint32_t& result) const {
        if (has_uniform_children(key, SVDAG::kEmpty)) {
            result = SVDAG::kEmpty;
            return true;
        }
        if (has_uniform_children(key, SVDAG::kFilled)) {
            result = SVDAG::kFilled;
            return true;
        }
        return false;
    }

    [[nodiscard]] uint32_t append(uint32_t level,
                                  const SVDAGBuildNode& node) {
        if (level >= dag_levels_.size()) {
            throw std::out_of_range("SVDAG: DAG level out of bounds");
        }
        SVDAG::PackedLevel& packed_level = dag_levels_[level];
        const uint32_t offset =
            SVDAG::checked_word_offset(packed_level.words.size());

        uint32_t child_mask = 0;
        uint32_t filled_mask = 0;
        std::array<uint32_t, SVDAG::kChildCount> child_ptrs = {};
        uint32_t child_ptr_count = 0;

        for (uint32_t child_index = 0; child_index < SVDAG::kChildCount;
             ++child_index) {
            const uint32_t child = node.child[child_index];
            if (child == SVDAG::kEmpty) {
                continue;
            }

            const uint32_t bit = SVDAG::mask_bit(child_index);
            child_mask |= bit;
            if (child == SVDAG::kFilled) {
                filled_mask |= bit;
                continue;
            }
            if (child < SVDAG::kFirstNode) {
                throw std::logic_error("SVDAG: invalid child pointer");
            }
            child_ptrs[child_ptr_count++] = child;
        }

        const uint32_t header = (child_mask << SVDAG::kChildMaskShift) |
                                (filled_mask << SVDAG::kFilledMaskShift);
        packed_level.words.push_back(header);
        packed_level.words.insert(packed_level.words.end(), child_ptrs.begin(),
                                  child_ptrs.begin() + child_ptr_count);
        ++packed_level.node_count;

        return SVDAG::kFirstNode + offset;
    }

    [[nodiscard]] Result finish(uint32_t max_level, uint32_t root) {
        return SVDAG(max_level, root, std::move(dag_levels_));
    }

  private:
    std::vector<SVDAG::PackedLevel> dag_levels_;
};

} // namespace detail

} // namespace algo::svt
