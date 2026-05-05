#pragma once

#include <algo/hash/murmur3.hpp>
#include <algo/memory/virtual_word_memory.hpp>
#include <algo/svt/constants.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace algo::svt {

class HashDAG {
  public:
    struct Box {
        uint32_t min_x = 0;
        uint32_t min_y = 0;
        uint32_t min_z = 0;
        uint32_t max_x = 0;
        uint32_t max_y = 0;
        uint32_t max_z = 0;
    };

    explicit HashDAG(uint32_t depth = DEFAULT_MAX_DEPTH) : depth_(depth) {
        validate_config();
        initialize_levels();
        initialize_full_subtrees();
    }

    void set_voxel(uint32_t x, uint32_t y, uint32_t z, bool value) {
        validate_coordinates(x, y, z);
        const Box box{x, y, z, x + 1, y + 1, z + 1};
        edit(EditOp{box, value ? EditKind::Fill : EditKind::Clear});
    }

    void fill_box(const Box& box) {
        edit(EditOp{box, EditKind::Fill});
    }

    void clear_box(const Box& box) {
        edit(EditOp{box, EditKind::Clear});
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        uint32_t ptr = root();
        if (ptr == kEmptyNode) {
            return false;
        }

        for (uint32_t level = 0; level < depth_; ++level) {
            const DecodedNode node = read_node(ptr);
            const uint32_t child_index = get_child_index(x, y, z, level);
            if ((node.child_mask & (uint8_t{1} << child_index)) == 0) {
                return false;
            }

            ptr = node.child_ptrs[child_index];
            if (level + 1 == depth_) {
                const uint64_t leaf = read_leaf(ptr);
                return (leaf & (uint64_t{1} << get_leaf_offset(x, y, z))) != 0;
            }
        }

        return false;
    }

    [[nodiscard]] uint32_t root() const {
        return root_;
    }

    [[nodiscard]] uint32_t depth() const {
        return depth_;
    }

    [[nodiscard]] uint32_t world_voxel_count() const {
        return world_voxel_count(depth_);
    }

    [[nodiscard]] static constexpr uint32_t world_voxel_count(uint32_t depth) {
        return LEAF_VOXEL_COUNT << depth;
    }

    [[nodiscard]] static constexpr uint32_t
    depth_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(
            requested_world_size, LEAF_VOXEL_COUNT, 1, 1,
            kMaxSupportedDepth);
    }

    [[nodiscard]] std::size_t node_count() const {
        std::size_t count = 0;
        for (const Level& level : node_levels_) {
            count += level.entry_count();
        }
        return count;
    }

    [[nodiscard]] std::size_t leaf_count() const {
        return leaf_level_.entry_count();
    }

    [[nodiscard]] std::size_t mapped_page_count() const {
        return memory_.mapped_page_count();
    }

    [[nodiscard]] std::size_t allocated_word_count() const {
        std::size_t count = leaf_level_.used_word_count();
        for (const Level& level : node_levels_) {
            count += level.used_word_count();
        }
        return count;
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        std::size_t bytes = memory_.memory_usage_bytes();
        bytes += leaf_level_.memory_usage_bytes();
        for (const Level& level : node_levels_) {
            bytes += level.memory_usage_bytes();
        }
        bytes += full_node_ptrs_.capacity() * sizeof(uint32_t);
        return bytes;
    }

    void collect_garbage() {
        std::vector<std::vector<uint32_t>> live_nodes(depth_);
        std::vector<uint32_t> current;
        append_non_empty(current, root_);
        if (!full_node_ptrs_.empty()) {
            append_non_empty(current, full_node_ptrs_[0]);
        }

        for (uint32_t level = 0; level < depth_; ++level) {
            sort_unique(current);
            live_nodes[level] = current;

            std::vector<uint32_t> next;
            for (const uint32_t ptr : current) {
                const DecodedNode node = read_node(ptr);
                for (uint32_t child = 0; child < kChildCount; ++child) {
                    if ((node.child_mask & (uint8_t{1} << child)) != 0) {
                        append_non_empty(next, node.child_ptrs[child]);
                    }
                }
            }
            current = std::move(next);
        }
        sort_unique(current);

        WordMemory new_memory;
        std::vector<Level> new_node_levels;
        Level new_leaf_level;
        initialize_levels(new_node_levels, new_leaf_level);

        auto child_mapping = compact_leaves(current, new_memory, new_leaf_level);
        const uint32_t new_full_leaf_ptr = mapped_ptr(child_mapping, full_leaf_ptr_);
        std::vector<uint32_t> new_full_node_ptrs(depth_, kEmptyNode);
        uint32_t new_root = root_;

        for (uint32_t reverse_level = depth_; reverse_level > 0; --reverse_level) {
            const uint32_t level = reverse_level - 1;
            auto node_mapping = compact_nodes(live_nodes[level],
                                              child_mapping,
                                              new_memory,
                                              new_node_levels[level]);

            if (root_ != kEmptyNode && level == 0) {
                new_root = mapped_ptr(node_mapping, root_);
            }
            if (full_node_ptrs_[level] != kEmptyNode) {
                new_full_node_ptrs[level] =
                    mapped_ptr(node_mapping, full_node_ptrs_[level]);
            }
            child_mapping = std::move(node_mapping);
        }

        memory_ = std::move(new_memory);
        node_levels_ = std::move(new_node_levels);
        leaf_level_ = std::move(new_leaf_level);
        root_ = new_root;
        full_leaf_ptr_ = new_full_leaf_ptr;
        full_node_ptrs_ = std::move(new_full_node_ptrs);
    }

  private:
    static constexpr uint32_t kChildCount = 8;
    static constexpr uint32_t kEmptyNode = 0;
    static constexpr uint64_t kLeafFullMask = ~uint64_t{0};
    static constexpr uint32_t kPageWords = 512;
    static constexpr uint32_t kBucketWordBits = 11;
    static constexpr uint32_t kNearRootBucketBits = 10;
    static constexpr uint32_t kLowerBucketBits = 16;
    static constexpr uint32_t kLowerBucketStartLevel = 10;
    static constexpr uint32_t kMaxSupportedDepth = 29;

    using WordMemory = algo::memory::VirtualWordMemory<kPageWords, uint32_t>;

    class BucketCapacityExceeded : public std::overflow_error {
      public:
        BucketCapacityExceeded()
            : std::overflow_error("HashDAG: bucket address range exhausted") {}
    };

    struct Level {
        uint32_t base_address = 0;
        uint32_t bucket_words = 0;
        std::vector<uint32_t> bucket_used_words;
        std::size_t entries = 0;

        Level() = default;

        Level(uint32_t base, uint32_t bucket_count, uint32_t bucket_word_count)
            : base_address(base),
              bucket_words(bucket_word_count),
              bucket_used_words(bucket_count, 0) {}

        [[nodiscard]] uint32_t bucket_count() const {
            return static_cast<uint32_t>(bucket_used_words.size());
        }

        [[nodiscard]] uint32_t bucket_start(uint32_t bucket) const {
            return base_address + bucket * bucket_words;
        }

        [[nodiscard]] std::size_t entry_count() const {
            return entries;
        }

        [[nodiscard]] std::size_t used_word_count() const {
            std::size_t count = 0;
            for (const uint32_t used : bucket_used_words) {
                count += used;
            }
            return count;
        }

        [[nodiscard]] std::size_t memory_usage_bytes() const {
            return bucket_used_words.capacity() * sizeof(uint32_t);
        }
    };

    struct DecodedNode {
        uint8_t child_mask = 0;
        std::array<uint32_t, kChildCount> child_ptrs = {};
    };

    enum class EditKind {
        Fill,
        Clear,
    };

    struct EditOp {
        Box box;
        EditKind kind = EditKind::Fill;
    };

    void validate_config() const {
        if (depth_ == 0 || depth_ > kMaxSupportedDepth) {
            throw std::out_of_range("HashDAG: depth out of supported range");
        }
        if (!std::has_single_bit(kPageWords) || kPageWords < 16) {
            throw std::out_of_range("HashDAG: page_words must be a power of two >= 16");
        }
        if (kBucketWordBits >= 31) {
            throw std::out_of_range("HashDAG: bucket_word_bits out of range");
        }
        if (kNearRootBucketBits >= 31 || kLowerBucketBits >= 31) {
            throw std::out_of_range("HashDAG: bucket bits out of range");
        }
        const uint32_t bucket_words = uint32_t{1} << kBucketWordBits;
        if (bucket_words < 9 || bucket_words < kPageWords) {
            throw std::out_of_range(
                "HashDAG: bucket must fit a node and at least one page");
        }
    }

    void initialize_levels() {
        initialize_levels(node_levels_, leaf_level_);
    }

    void initialize_levels(std::vector<Level>& node_levels, Level& leaf_level) const {
        const uint32_t bucket_words = uint32_t{1} << kBucketWordBits;
        // Address 0 is reserved as kEmptyNode, so real entries start at 1.
        uint64_t base = 1;

        node_levels.clear();
        node_levels.reserve(depth_);
        for (uint32_t level = 0; level < depth_; ++level) {
            const uint32_t bucket_bits = bucket_bits_for_level(level);
            const uint32_t bucket_count = uint32_t{1} << bucket_bits;
            const uint64_t level_words =
                uint64_t{bucket_count} * uint64_t{bucket_words};
            ensure_address_range(base, level_words);
            node_levels.emplace_back(static_cast<uint32_t>(base),
                                     bucket_count,
                                     bucket_words);
            base += level_words;
        }

        const uint32_t leaf_bucket_bits = bucket_bits_for_level(depth_);
        const uint32_t leaf_bucket_count = uint32_t{1} << leaf_bucket_bits;
        const uint64_t leaf_level_words =
            uint64_t{leaf_bucket_count} * uint64_t{bucket_words};
        ensure_address_range(base, leaf_level_words);
        leaf_level =
            Level(static_cast<uint32_t>(base), leaf_bucket_count, bucket_words);
    }

    [[nodiscard]] uint32_t bucket_bits_for_level(uint32_t level) const {
        return level >= kLowerBucketStartLevel ? kLowerBucketBits
                                               : kNearRootBucketBits;
    }

    static void ensure_address_range(uint64_t base, uint64_t word_count) {
        if (word_count == 0 ||
            base + word_count - 1 > std::numeric_limits<uint32_t>::max() - 1ull) {
            throw std::out_of_range("HashDAG: virtual address space exhausted");
        }
    }

    void initialize_full_subtrees() {
        full_leaf_ptr_ = find_or_append_leaf(kLeafFullMask);
        full_node_ptrs_.assign(depth_, kEmptyNode);

        uint32_t child_ptr = full_leaf_ptr_;
        for (uint32_t remaining = depth_; remaining > 0; --remaining) {
            const uint32_t level = remaining - 1;
            std::array<uint32_t, kChildCount> child_ptrs = {};
            child_ptrs.fill(child_ptr);
            child_ptr = find_or_append_node(level, 0xffu, child_ptrs);
            full_node_ptrs_[level] = child_ptr;
        }
    }

    void edit(const EditOp& op) {
        validate_box(op.box);
        if (op.box.min_x == op.box.max_x || op.box.min_y == op.box.max_y ||
            op.box.min_z == op.box.max_z) {
            return;
        }

        bool retried_after_gc = false;
        for (;;) {
            try {
                const uint32_t new_root =
                    edit_node(op, 0, 0, 0, 0, world_voxel_count(), root());
                if (new_root != root_) {
                    root_ = new_root;
                }
                return;
            } catch (const BucketCapacityExceeded&) {
                if (retried_after_gc) {
                    throw;
                }
                retried_after_gc = true;
                collect_garbage();
            }
        }
    }

    [[nodiscard]] uint32_t edit_node(const EditOp& op,
                                     uint32_t level,
                                     uint32_t origin_x,
                                     uint32_t origin_y,
                                     uint32_t origin_z,
                                     uint32_t extent,
                                     uint32_t node_ptr) {
        const Box volume{origin_x,
                         origin_y,
                         origin_z,
                         origin_x + extent,
                         origin_y + extent,
                         origin_z + extent};
        if (!intersects(op.box, volume)) {
            return node_ptr;
        }

        if (contains(op.box, volume)) {
            return op.kind == EditKind::Fill ? full_node_ptrs_[level] : kEmptyNode;
        }

        if (level + 1 == depth_) {
            return edit_leaf_parent(op, level, origin_x, origin_y, origin_z, node_ptr);
        }

        DecodedNode node;
        if (node_ptr != kEmptyNode) {
            node = read_node(node_ptr);
        }

        bool any_child_changed = false;
        const uint32_t child_extent = extent / 2;
        for (uint32_t child = 0; child < kChildCount; ++child) {
            const uint32_t child_x =
                origin_x + ((child & 1u) != 0 ? child_extent : 0);
            const uint32_t child_y =
                origin_y + ((child & 2u) != 0 ? child_extent : 0);
            const uint32_t child_z =
                origin_z + ((child & 4u) != 0 ? child_extent : 0);

            const uint32_t old_child =
                (node.child_mask & (uint8_t{1} << child)) != 0
                    ? node.child_ptrs[child]
                    : kEmptyNode;
            const uint32_t new_child = edit_node(op,
                                                 level + 1,
                                                 child_x,
                                                 child_y,
                                                 child_z,
                                                 child_extent,
                                                 old_child);
            if (new_child == old_child) {
                continue;
            }

            any_child_changed = true;
            if (new_child == kEmptyNode) {
                node.child_mask &= static_cast<uint8_t>(~(uint8_t{1} << child));
                node.child_ptrs[child] = kEmptyNode;
            } else {
                node.child_mask |= static_cast<uint8_t>(uint8_t{1} << child);
                node.child_ptrs[child] = new_child;
            }
        }

        if (!any_child_changed) {
            return node_ptr;
        }
        if (node.child_mask == 0) {
            return kEmptyNode;
        }
        const uint32_t new_node =
            find_or_append_node(level, node.child_mask, node.child_ptrs);
        return new_node == node_ptr ? node_ptr : new_node;
    }

    [[nodiscard]] uint32_t edit_leaf_parent(const EditOp& op,
                                            uint32_t level,
                                            uint32_t origin_x,
                                            uint32_t origin_y,
                                            uint32_t origin_z,
                                            uint32_t node_ptr) {
        DecodedNode node;
        if (node_ptr != kEmptyNode) {
            node = read_node(node_ptr);
        }

        bool any_child_changed = false;
        for (uint32_t child = 0; child < kChildCount; ++child) {
            const uint32_t child_x =
                origin_x + ((child & 1u) != 0 ? LEAF_VOXEL_COUNT : 0);
            const uint32_t child_y =
                origin_y + ((child & 2u) != 0 ? LEAF_VOXEL_COUNT : 0);
            const uint32_t child_z =
                origin_z + ((child & 4u) != 0 ? LEAF_VOXEL_COUNT : 0);
            const Box leaf_volume{child_x,
                                  child_y,
                                  child_z,
                                  child_x + LEAF_VOXEL_COUNT,
                                  child_y + LEAF_VOXEL_COUNT,
                                  child_z + LEAF_VOXEL_COUNT};

            if (!intersects(op.box, leaf_volume)) {
                continue;
            }

            const bool had_child = (node.child_mask & (uint8_t{1} << child)) != 0;
            const uint32_t old_ptr = had_child ? node.child_ptrs[child] : kEmptyNode;
            uint32_t new_ptr = old_ptr;

            if (contains(op.box, leaf_volume)) {
                new_ptr = op.kind == EditKind::Fill ? full_leaf_ptr_ : kEmptyNode;
            } else {
                uint64_t leaf_data = old_ptr == kEmptyNode ? 0 : read_leaf(old_ptr);
                leaf_data = edit_leaf_bits(op, leaf_volume, leaf_data);
                new_ptr = find_or_append_leaf(leaf_data);
            }

            if (new_ptr == old_ptr) {
                continue;
            }

            any_child_changed = true;
            if (new_ptr == kEmptyNode) {
                node.child_mask &= static_cast<uint8_t>(~(uint8_t{1} << child));
                node.child_ptrs[child] = kEmptyNode;
            } else {
                node.child_mask |= static_cast<uint8_t>(uint8_t{1} << child);
                node.child_ptrs[child] = new_ptr;
            }
        }

        if (!any_child_changed) {
            return node_ptr;
        }
        if (node.child_mask == 0) {
            return kEmptyNode;
        }
        const uint32_t new_node =
            find_or_append_node(level, node.child_mask, node.child_ptrs);
        return new_node == node_ptr ? node_ptr : new_node;
    }

    [[nodiscard]] static uint64_t edit_leaf_bits(const EditOp& op,
                                                 const Box& leaf_volume,
                                                 uint64_t leaf_data) {
        for (uint32_t z = leaf_volume.min_z; z < leaf_volume.max_z; ++z) {
            for (uint32_t y = leaf_volume.min_y; y < leaf_volume.max_y; ++y) {
                for (uint32_t x = leaf_volume.min_x; x < leaf_volume.max_x; ++x) {
                    if (x < op.box.min_x || x >= op.box.max_x || y < op.box.min_y ||
                        y >= op.box.max_y || z < op.box.min_z || z >= op.box.max_z) {
                        continue;
                    }
                    const uint32_t bit = leaf_offset_in_volume(x, y, z, leaf_volume);
                    const uint64_t mask = uint64_t{1} << bit;
                    if (op.kind == EditKind::Fill) {
                        leaf_data |= mask;
                    } else {
                        leaf_data &= ~mask;
                    }
                }
            }
        }
        return leaf_data;
    }

    [[nodiscard]] uint32_t find_or_append_node(
        uint32_t level,
        uint8_t child_mask,
        const std::array<uint32_t, kChildCount>& child_ptrs) {
        const std::vector<uint32_t> words = encode_node(child_mask, child_ptrs);

        Level& store = node_levels_[level];
        const uint32_t bucket =
            algo::hash::murmur3_words(words) & (store.bucket_count() - 1u);
        return find_or_append_entry(memory_, store, bucket, words, false);
    }

    [[nodiscard]] uint32_t find_or_append_leaf(uint64_t leaf_data) {
        if (leaf_data == 0) {
            return kEmptyNode;
        }

        std::vector<uint32_t> words{
            static_cast<uint32_t>(leaf_data),
            static_cast<uint32_t>(leaf_data >> 32),
        };

        const uint32_t bucket = algo::hash::murmur3_u64(leaf_data) &
                                (leaf_level_.bucket_count() - 1u);
        return find_or_append_entry(memory_, leaf_level_, bucket, words, true);
    }

    [[nodiscard]] static uint32_t find_entry(const WordMemory& memory,
                                             const Level& level,
                                             uint32_t bucket,
                                             const std::vector<uint32_t>& words,
                                             bool fixed_size_entries) {
        const uint32_t entry_words = static_cast<uint32_t>(words.size());
        uint32_t offset = 0;
        const uint32_t used_words = level.bucket_used_words[bucket];
        const uint32_t start = level.bucket_start(bucket);

        while (offset < used_words) {
            const uint32_t address = start + offset;
            const uint32_t page_offset = address % kPageWords;
            const uint32_t remaining_page_words = kPageWords - page_offset;

            uint32_t current_entry_words = entry_words;
            if (!fixed_size_entries) {
                const uint32_t child_mask = memory.read(address) & 0xffu;
                if (child_mask == 0) {
                    offset += remaining_page_words;
                    continue;
                }
                current_entry_words = 1 + std::popcount(child_mask);
            }

            if (remaining_page_words < current_entry_words) {
                offset += remaining_page_words;
                continue;
            }

            bool equal = true;
            if (current_entry_words != entry_words) {
                equal = false;
            } else {
                for (uint32_t i = 0; i < entry_words; ++i) {
                    if (memory.read(address + i) != words[i]) {
                        equal = false;
                        break;
                    }
                }
            }
            if (equal) {
                return address;
            }

            if (fixed_size_entries) {
                offset += entry_words;
            } else {
                offset += current_entry_words;
            }
        }

        return kEmptyNode;
    }

    [[nodiscard]] static uint32_t append_entry(WordMemory& memory,
                                               Level& level,
                                               uint32_t bucket,
                                               const std::vector<uint32_t>& words) {
        const uint32_t entry_words = static_cast<uint32_t>(words.size());
        uint32_t used_words = level.bucket_used_words[bucket];
        uint32_t address = level.bucket_start(bucket) + used_words;
        const uint32_t page_offset = address % kPageWords;
        const uint32_t remaining_page_words = kPageWords - page_offset;

        if (remaining_page_words < entry_words) {
            used_words += remaining_page_words;
            address += remaining_page_words;
        }

        if (uint64_t{used_words} + entry_words > level.bucket_words) {
            throw BucketCapacityExceeded();
        }

        for (uint32_t i = 0; i < entry_words; ++i) {
            memory.write(address + i, words[i]);
        }
        level.bucket_used_words[bucket] = used_words + entry_words;
        ++level.entries;
        return address;
    }

    [[nodiscard]] static uint32_t find_or_append_entry(
        WordMemory& memory,
        Level& level,
        uint32_t bucket,
        const std::vector<uint32_t>& words,
        bool fixed_size_entries) {
        const uint32_t found =
            find_entry(memory, level, bucket, words, fixed_size_entries);
        if (found != kEmptyNode) {
            return found;
        }
        return append_entry(memory, level, bucket, words);
    }

    [[nodiscard]] static std::vector<uint32_t> encode_node(
        uint8_t child_mask,
        const std::array<uint32_t, kChildCount>& child_ptrs) {
        std::vector<uint32_t> words;
        words.reserve(1 + kChildCount);
        words.push_back(child_mask);
        for (uint32_t child = 0; child < kChildCount; ++child) {
            if ((child_mask & (uint8_t{1} << child)) != 0) {
                words.push_back(child_ptrs[child]);
            }
        }
        return words;
    }

    [[nodiscard]] static std::vector<uint32_t> encode_leaf(uint64_t leaf_data) {
        return {
            static_cast<uint32_t>(leaf_data),
            static_cast<uint32_t>(leaf_data >> 32),
        };
    }

    static void append_non_empty(std::vector<uint32_t>& ptrs, uint32_t ptr) {
        if (ptr != kEmptyNode) {
            ptrs.push_back(ptr);
        }
    }

    static void sort_unique(std::vector<uint32_t>& ptrs) {
        std::sort(ptrs.begin(), ptrs.end());
        ptrs.erase(std::unique(ptrs.begin(), ptrs.end()), ptrs.end());
    }

    [[nodiscard]] static uint32_t mapped_ptr(
        const std::unordered_map<uint32_t, uint32_t>& mapping,
        uint32_t old_ptr) {
        if (old_ptr == kEmptyNode) {
            return kEmptyNode;
        }
        const auto it = mapping.find(old_ptr);
        if (it == mapping.end()) {
            throw std::logic_error("HashDAG: missing garbage collection mapping");
        }
        return it->second;
    }

    [[nodiscard]] std::unordered_map<uint32_t, uint32_t> compact_leaves(
        const std::vector<uint32_t>& live_leaves,
        WordMemory& new_memory,
        Level& new_leaf_level) const {
        std::unordered_map<uint32_t, uint32_t> mapping;
        mapping.reserve(live_leaves.size());

        for (const uint32_t old_ptr : live_leaves) {
            const uint64_t leaf_data = read_leaf(old_ptr);
            const std::vector<uint32_t> words = encode_leaf(leaf_data);
            const uint32_t bucket = algo::hash::murmur3_u64(leaf_data) &
                                    (new_leaf_level.bucket_count() - 1u);
            const uint32_t new_ptr =
                find_or_append_entry(new_memory, new_leaf_level, bucket, words, true);
            mapping.emplace(old_ptr, new_ptr);
        }
        return mapping;
    }

    [[nodiscard]] std::unordered_map<uint32_t, uint32_t> compact_nodes(
        const std::vector<uint32_t>& live_nodes,
        const std::unordered_map<uint32_t, uint32_t>& child_mapping,
        WordMemory& new_memory,
        Level& new_level) const {
        std::unordered_map<uint32_t, uint32_t> mapping;
        mapping.reserve(live_nodes.size());

        for (const uint32_t old_ptr : live_nodes) {
            DecodedNode node = read_node(old_ptr);
            for (uint32_t child = 0; child < kChildCount; ++child) {
                if ((node.child_mask & (uint8_t{1} << child)) != 0) {
                    node.child_ptrs[child] =
                        mapped_ptr(child_mapping, node.child_ptrs[child]);
                }
            }

            const std::vector<uint32_t> words =
                encode_node(node.child_mask, node.child_ptrs);
            const uint32_t bucket =
                algo::hash::murmur3_words(words) & (new_level.bucket_count() - 1u);
            const uint32_t new_ptr =
                find_or_append_entry(new_memory, new_level, bucket, words, false);
            mapping.emplace(old_ptr, new_ptr);
        }
        return mapping;
    }

    [[nodiscard]] DecodedNode read_node(uint32_t ptr) const {
        DecodedNode node;
        node.child_mask = static_cast<uint8_t>(memory_.read(ptr) & 0xffu);

        uint32_t word_offset = 1;
        for (uint32_t child = 0; child < kChildCount; ++child) {
            if ((node.child_mask & (uint8_t{1} << child)) == 0) {
                continue;
            }
            node.child_ptrs[child] = memory_.read(ptr + word_offset);
            ++word_offset;
        }
        return node;
    }

    [[nodiscard]] uint64_t read_leaf(uint32_t ptr) const {
        const uint64_t lo = memory_.read(ptr);
        const uint64_t hi = memory_.read(ptr + 1);
        return lo | (hi << 32);
    }

    [[nodiscard]] uint32_t get_child_index(uint32_t x,
                                           uint32_t y,
                                           uint32_t z,
                                           uint32_t level) const {
        const uint32_t scale = LEAF_VOXEL_COUNT << (depth_ - level - 1);
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

    [[nodiscard]] static uint32_t leaf_offset_in_volume(uint32_t x,
                                                        uint32_t y,
                                                        uint32_t z,
                                                        const Box& leaf_volume) {
        const uint32_t x_offset = x - leaf_volume.min_x;
        const uint32_t y_offset = y - leaf_volume.min_y;
        const uint32_t z_offset = z - leaf_volume.min_z;
        return x_offset + y_offset * LEAF_VOXEL_COUNT +
               z_offset * LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT;
    }

    [[nodiscard]] static bool intersects(const Box& a, const Box& b) {
        return a.min_x < b.max_x && a.max_x > b.min_x && a.min_y < b.max_y &&
               a.max_y > b.min_y && a.min_z < b.max_z && a.max_z > b.min_z;
    }

    [[nodiscard]] static bool contains(const Box& outer, const Box& inner) {
        return outer.min_x <= inner.min_x && outer.min_y <= inner.min_y &&
               outer.min_z <= inner.min_z && outer.max_x >= inner.max_x &&
               outer.max_y >= inner.max_y && outer.max_z >= inner.max_z;
    }

    void validate_coordinates(uint32_t x, uint32_t y, uint32_t z) const {
        if (x >= world_voxel_count() || y >= world_voxel_count() ||
            z >= world_voxel_count()) {
            throw std::out_of_range("HashDAG: voxel coordinates out of bounds");
        }
    }

    void validate_box(const Box& box) const {
        if (box.min_x > box.max_x || box.min_y > box.max_y ||
            box.min_z > box.max_z || box.max_x > world_voxel_count() ||
            box.max_y > world_voxel_count() || box.max_z > world_voxel_count()) {
            throw std::out_of_range("HashDAG: box out of bounds");
        }
    }

    uint32_t depth_;
    WordMemory memory_;
    std::vector<Level> node_levels_;
    Level leaf_level_;
    uint32_t full_leaf_ptr_ = kEmptyNode;
    std::vector<uint32_t> full_node_ptrs_;
    uint32_t root_ = kEmptyNode;
};

} // namespace algo::svt
