#pragma once

#include <algo/svt/constants.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace algo::svt {

inline constexpr uint32_t kDefaultPackedForwardingLeafPageBits = 15;

template <uint32_t PageBits = kDefaultPackedForwardingLeafPageBits>
class SVOForwardingPackedLeafImpl {
  public:
    explicit SVOForwardingPackedLeafImpl(uint32_t max_depth = DEFAULT_MAX_DEPTH)
        : max_depth_(max_depth) {
        validate_max_depth();
        pages_.push_back(Page{});
        mark_group_used(pages_[0], GroupIndex{0});
        used_group_count_ = 1;
        init_node_cell(root_node_index(), NodeCell{});
    }

    void set_voxel(uint32_t x, uint32_t y, uint32_t z, bool value) {
        validate_coordinates(x, y, z);

        auto [node_index, node] = resolve_node(root_cell_index());
        std::vector<NodeIndex> node_indices;
        std::vector<bool> node_relocated;
        std::vector<uint32_t> child_indices;
        node_indices.reserve(max_depth_ + 1);
        node_relocated.reserve(max_depth_ + 1);
        child_indices.reserve(max_depth_ + 1);
        node_indices.push_back(node_index);
        node_relocated.push_back(node_index.raw() != root_cell_index().raw());

        for (uint32_t depth = 0; depth < max_depth_; ++depth) {
            const uint32_t child_index = get_child_index(x, y, z, depth);
            child_indices.push_back(child_index);

            if (has_child(node, child_index)) {
                const CellIndex child_cell =
                    get_child_node_cell_index(node_index, node, child_index);
                auto [next_index, next_node] = resolve_node(child_cell);
                node_index = next_index;
                node = next_node;
                node_indices.push_back(node_index);
                node_relocated.push_back(next_index.raw() != child_cell.raw());
                continue;
            }

            const bool filled = is_filled(node, child_index);
            if (filled == value) {
                return;
            }

            const NodeIndex original_node_index = node_index;
            node_index = ensure_node_group(node_index);
            node_indices.back() = node_index;
            node_relocated.back() = node_index.raw() != original_node_index.raw();

            node = get_node_cell(node_index);
            set_child_mask(node, child_index, true);
            set_node_cell(node_index, node);

            const NodeIndex child_node_index =
                node_index_from_cell(get_child_node_cell_index(node_index, node, child_index));
            set_node_cell(child_node_index, make_uniform_node(filled));
            node_index = child_node_index;
            node = get_node_cell(node_index);
            node_indices.push_back(node_index);
            node_relocated.push_back(false);
        }

        const uint32_t child_index = get_child_index(x, y, z, max_depth_);
        child_indices.push_back(child_index);

        if (!has_child(node, child_index)) {
            const bool filled = is_filled(node, child_index);
            if (filled == value) {
                return;
            }

            const NodeIndex original_node_index = node_index;
            node_index = ensure_leaf_block(node_index);
            node_indices.back() = node_index;
            node_relocated.back() = node_index.raw() != original_node_index.raw();

            node = get_node_cell(node_index);
            set_child_mask(node, child_index, true);
            set_node_cell(node_index, node);
        }

        const CellIndex leaf_cell = get_leaf_cell_index(node_index, get_node_cell(node_index), child_index);
        uint64_t leaf = get_leaf_value(leaf_cell);
        const uint32_t bit_offset = get_leaf_offset(x, y, z);
        const uint64_t mask = uint64_t{1} << bit_offset;
        const bool old_value = (leaf & mask) != 0;
        if (old_value == value) {
            return;
        }

        if (value) {
            leaf |= mask;
        } else {
            leaf &= ~mask;
        }
        set_leaf_value(leaf_cell, leaf);

        NodeCell leaf_parent = get_node_cell(node_index);
        if (leaf == 0) {
            set_child_mask(leaf_parent, child_index, false);
            set_filled(leaf_parent, child_index, false);
            set_node_cell(node_index, leaf_parent);
        } else if (leaf == kLeafFullMask) {
            set_child_mask(leaf_parent, child_index, false);
            set_filled(leaf_parent, child_index, true);
            set_node_cell(node_index, leaf_parent);
        }

        collapse_path(node_indices, node_relocated, child_indices);
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        auto [node_index, node] = resolve_node(root_cell_index());
        for (uint32_t depth = 0; depth < max_depth_; ++depth) {
            const uint32_t child_index = get_child_index(x, y, z, depth);
            if (!has_child(node, child_index)) {
                return is_filled(node, child_index);
            }

            auto [next_index, next_node] =
                resolve_node(get_child_node_cell_index(node_index, node, child_index));
            node_index = next_index;
            node = next_node;
        }

        const uint32_t child_index = get_child_index(x, y, z, max_depth_);
        if (!has_child(node, child_index)) {
            return is_filled(node, child_index);
        }

        const uint64_t leaf =
            get_leaf_value(get_leaf_cell_index(node_index, node, child_index));
        return (leaf & (uint64_t{1} << get_leaf_offset(x, y, z))) != 0;
    }

    [[nodiscard]] std::size_t node_count() const {
        return used_group_count_ * kGroupSize;
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

    [[nodiscard]] uint32_t page_count() const {
        return static_cast<uint32_t>(pages_.size());
    }

    [[nodiscard]] std::size_t forwarded_cell_count() const {
        return forwarding_cell_count_;
    }

    [[nodiscard]] static constexpr uint32_t world_voxel_count(uint32_t max_depth) {
        return (LEAF_VOXEL_COUNT << 1u) << max_depth;
    }

    [[nodiscard]] static constexpr uint32_t
    max_depth_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(
            requested_world_size, LEAF_VOXEL_COUNT << 1u, 1, 0,
            kMaxSupportedDepth);
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        return pages_.capacity() * sizeof(Page);
    }

    [[nodiscard]] std::size_t node_storage_bytes() const {
        return node_count() * sizeof(NodeCell);
    }

  private:
    static constexpr uint32_t kChildCount = 8;
    static constexpr uint32_t kGroupSize = 9;
    static constexpr uint32_t kNodeGroupCells = kGroupSize;
    static constexpr uint32_t kRelocatedNodeSlot = kChildCount;
    static constexpr uint32_t kLeafBlockGroups = 2;
    static constexpr uint32_t kLeafBlockCells = kGroupSize * kLeafBlockGroups;
    static constexpr uint32_t kLeafForwardingSlot = 0;
    static constexpr uint32_t kLeafDataStartSlot = 1;
    static constexpr uint32_t kLeafUnusedSlot = kLeafBlockCells - 1;
    static constexpr uint32_t kLeafWordsPerLeaf = 2;
    static constexpr uint32_t kRootNodeIndex = 0;
    static constexpr uint32_t kForwardingTagBit = 1u << 31;
    static constexpr uint32_t kChildMaskMask = 0x000000FFu;
    static constexpr uint32_t kFilledMaskMask = 0x0000FF00u;
    static constexpr uint32_t kLocalIndexMask = 0x7FFF0000u;
    static constexpr uint32_t kFilledShift = 8;
    static constexpr uint32_t kLocalIndexShift = 16;
    static constexpr uint32_t kForwardingIndexMask = 0x7FFFFFFFu;
    static constexpr uint64_t kLeafFullMask = ~uint64_t{0};
    static constexpr uint32_t kMaxSupportedDepth = 28;
    static constexpr uint32_t kPageBits = PageBits;
    static constexpr uint32_t kGroupsPerPage = (1u << kPageBits) / kGroupSize;
    static constexpr uint32_t kFreeGroupBitmapWords =
        (kGroupsPerPage + 63u) / 64u;
    static_assert(kPageBits >= 7 && kPageBits <= kDefaultPackedForwardingLeafPageBits,
                  "PageBits out of supported range");
    static_assert(kGroupsPerPage > 1, "Page must have room for at least two groups");

    struct PageIndex final {
        explicit constexpr PageIndex(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct CellOffset final {
        explicit constexpr CellOffset(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct GroupIndex final {
        explicit constexpr GroupIndex(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct CellIndex final {
        explicit constexpr CellIndex(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct NodeIndex final {
        explicit constexpr NodeIndex(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct NodeCell {
        uint32_t value = 0;

        [[nodiscard]] uint8_t child_mask() const {
            return static_cast<uint8_t>(value & kChildMaskMask);
        }

        [[nodiscard]] uint8_t filled_mask() const {
            return static_cast<uint8_t>((value & kFilledMaskMask) >> kFilledShift);
        }

        [[nodiscard]] GroupIndex child_group_index() const {
            return GroupIndex{(value & kLocalIndexMask) >> kLocalIndexShift};
        }

        void set_filled_mask_byte(uint8_t mask) {
            value = (value & ~kFilledMaskMask) | (uint32_t(mask) << kFilledShift);
        }

        void set_child_group_index(GroupIndex idx) {
            value = (value & ~kLocalIndexMask) | (idx.raw() << kLocalIndexShift);
        }
    };

    static_assert(sizeof(PageIndex) == sizeof(uint32_t));
    static_assert(sizeof(CellOffset) == sizeof(uint32_t));
    static_assert(sizeof(GroupIndex) == sizeof(uint32_t));
    static_assert(sizeof(CellIndex) == sizeof(uint32_t));
    static_assert(sizeof(NodeIndex) == sizeof(uint32_t));
    static_assert(std::is_trivially_copyable_v<PageIndex>);
    static_assert(std::is_trivially_copyable_v<CellOffset>);
    static_assert(std::is_trivially_copyable_v<GroupIndex>);
    static_assert(std::is_trivially_copyable_v<CellIndex>);
    static_assert(std::is_trivially_copyable_v<NodeIndex>);

    struct Page {
        Page() : free_group_count(static_cast<uint16_t>(kGroupsPerPage)) {
            free_group_bitmap.fill(~uint64_t{0});
            if constexpr ((kGroupsPerPage % 64u) != 0) {
                free_group_bitmap.back() =
                    (uint64_t{1} << (kGroupsPerPage % 64u)) - 1u;
            }
        }

        std::array<uint32_t, 1u << kPageBits> cells{};
        std::array<uint64_t, kFreeGroupBitmapWords> free_group_bitmap{};
        uint16_t free_group_count = 0;
    };

    [[nodiscard]] constexpr uint32_t cells_per_page() const {
        return 1u << kPageBits;
    }

    [[nodiscard]] static CellIndex root_cell_index() {
        return CellIndex{kRootNodeIndex};
    }

    [[nodiscard]] static NodeIndex root_node_index() {
        return NodeIndex{kRootNodeIndex};
    }

    [[nodiscard]] static NodeIndex node_index_from_cell(CellIndex cell_index) {
        return NodeIndex{cell_index.raw()};
    }

    [[nodiscard]] static CellIndex cell_index_from_node(NodeIndex node_index) {
        return CellIndex{node_index.raw()};
    }

    [[nodiscard]] static uint32_t pack_forwarding(CellIndex target_index) {
        return kForwardingTagBit | (target_index.raw() & kForwardingIndexMask);
    }

    [[nodiscard]] static bool is_forwarding_cell_value(uint32_t cell) {
        return (cell & kForwardingTagBit) != 0;
    }

    [[nodiscard]] uint32_t get_cell_value(CellIndex absolute_index) const {
        const auto [page_index, cell_offset] = split_index(absolute_index);
        return pages_[page_index.raw()].cells[cell_offset.raw()];
    }

    void initialize_cell_value(CellIndex absolute_index, uint32_t value) {
        const auto [page_index, cell_offset] = split_index(absolute_index);
        pages_[page_index.raw()].cells[cell_offset.raw()] = value;
        if (is_forwarding_cell_value(value)) {
            ++forwarding_cell_count_;
        }
    }

    void update_cell_value(CellIndex absolute_index, uint32_t value) {
        const auto [page_index, cell_offset] = split_index(absolute_index);
        Page& page = pages_[page_index.raw()];
        const uint32_t old_value = page.cells[cell_offset.raw()];
        if (old_value == value) {
            return;
        }
        if (is_forwarding_cell_value(old_value)) {
            --forwarding_cell_count_;
        }
        if (is_forwarding_cell_value(value)) {
            ++forwarding_cell_count_;
        }
        page.cells[cell_offset.raw()] = value;
    }

    [[nodiscard]] NodeCell get_node_cell(NodeIndex node_index) const {
        return NodeCell{get_cell_value(cell_index_from_node(node_index))};
    }

    void init_node_cell(NodeIndex node_index, const NodeCell& node) {
        initialize_cell_value(cell_index_from_node(node_index), node.value);
    }

    void set_node_cell(NodeIndex node_index, const NodeCell& node) {
        update_cell_value(cell_index_from_node(node_index), node.value);
    }

    void set_forwarding_cell(CellIndex absolute_index, CellIndex target_index) {
        update_cell_value(absolute_index, pack_forwarding(target_index));
    }

    [[nodiscard]] std::pair<NodeIndex, NodeCell> resolve_node(CellIndex absolute_index) const {
        CellIndex current = absolute_index;
        uint32_t cell = get_cell_value(current);
        while (is_forwarding_cell_value(cell)) {
            current = CellIndex{cell & kForwardingIndexMask};
            cell = get_cell_value(current);
        }
        return {node_index_from_cell(current), NodeCell{cell}};
    }

    [[nodiscard]] static bool has_child(const NodeCell& node, uint32_t child_index) {
        return (node.child_mask() & (uint8_t{1} << child_index)) != 0;
    }

    [[nodiscard]] static bool is_filled(const NodeCell& node, uint32_t child_index) {
        return (node.filled_mask() & (uint8_t{1} << child_index)) != 0;
    }

    static void set_child_mask(NodeCell& node, uint32_t child_index, bool value) {
        const uint32_t bit = uint32_t{1} << child_index;
        if (value) {
            node.value |= bit;
        } else {
            node.value &= ~bit;
        }
    }

    static void set_filled(NodeCell& node, uint32_t child_index, bool value) {
        const uint32_t bit = uint32_t{1} << (child_index + kFilledShift);
        if (value) {
            node.value |= bit;
        } else {
            node.value &= ~bit;
        }
    }

    [[nodiscard]] static NodeCell make_uniform_node(bool filled) {
        NodeCell node{};
        node.set_filled_mask_byte(filled ? 0xFFu : 0u);
        return node;
    }

    [[nodiscard]] GroupIndex group_index_from_cell_offset(CellOffset cell_offset) const {
        return GroupIndex{cell_offset.raw() / kGroupSize};
    }

    [[nodiscard]] CellIndex group_start_index(PageIndex page_index,
                                              GroupIndex group_index) const {
        return CellIndex{
            page_index.raw() * cells_per_page() + group_index.raw() * kGroupSize};
    }

    [[nodiscard]] CellIndex get_block_start_index(NodeIndex node_index,
                                                  const NodeCell& node) const {
        const PageIndex page_index{node_index.raw() / cells_per_page()};
        return group_start_index(page_index, node.child_group_index());
    }

    [[nodiscard]] CellIndex get_child_node_cell_index(NodeIndex node_index,
                                                      const NodeCell& node,
                                                      uint32_t child_index) const {
        return CellIndex{get_block_start_index(node_index, node).raw() + child_index};
    }

    [[nodiscard]] CellIndex get_leaf_cell_index(NodeIndex node_index,
                                                const NodeCell& node,
                                                uint32_t child_index) const {
        return CellIndex{get_block_start_index(node_index, node).raw() +
                         kLeafDataStartSlot + child_index * kLeafWordsPerLeaf};
    }

    [[nodiscard]] NodeIndex relocated_node_index(CellIndex group_start) const {
        return node_index_from_cell(CellIndex{group_start.raw() + kRelocatedNodeSlot});
    }

    [[nodiscard]] NodeIndex relocated_leaf_parent_index(CellIndex block_start) const {
        return node_index_from_cell(CellIndex{block_start.raw() + kLeafForwardingSlot});
    }

    [[nodiscard]] uint64_t get_leaf_value(CellIndex leaf_cell) const {
        const uint64_t lo = get_cell_value(leaf_cell);
        const uint64_t hi = get_cell_value(CellIndex{leaf_cell.raw() + 1});
        return lo | (hi << 32);
    }

    void set_leaf_value(CellIndex leaf_cell, uint64_t value) {
        update_cell_value(leaf_cell, static_cast<uint32_t>(value & 0xFFFFFFFFu));
        update_cell_value(CellIndex{leaf_cell.raw() + 1},
                          static_cast<uint32_t>(value >> 32));
    }

    void init_leaf_value(CellIndex leaf_cell, uint64_t value) {
        initialize_cell_value(leaf_cell, static_cast<uint32_t>(value & 0xFFFFFFFFu));
        initialize_cell_value(CellIndex{leaf_cell.raw() + 1},
                              static_cast<uint32_t>(value >> 32));
    }

    [[nodiscard]] CellIndex allocate_group_in_page(PageIndex page_index) {
        Page& page = pages_[page_index.raw()];
        const GroupIndex group_index = take_free_group(page);
        ++used_group_count_;
        return group_start_index(page_index, group_index);
    }

    [[nodiscard]] CellIndex allocate_leaf_block_in_page(PageIndex page_index) {
        Page& page = pages_[page_index.raw()];
        const GroupIndex group_index = take_two_free_groups(page);
        used_group_count_ += kLeafBlockGroups;
        return group_start_index(page_index, group_index);
    }

    [[nodiscard]] CellIndex find_relocation_node_group() {
        for (uint32_t raw_page_index = 0; raw_page_index < pages_.size();
             ++raw_page_index) {
            const PageIndex page_index{raw_page_index};
            if (pages_[page_index.raw()].free_group_count != 0) {
                return allocate_group_in_page(page_index);
            }
        }

        pages_.push_back(Page{});
        validate_total_cell_capacity();
        return allocate_group_in_page(PageIndex{static_cast<uint32_t>(pages_.size() - 1)});
    }

    [[nodiscard]] CellIndex find_relocation_leaf_block() {
        for (uint32_t raw_page_index = 0; raw_page_index < pages_.size();
             ++raw_page_index) {
            const PageIndex page_index{raw_page_index};
            if (pages_[page_index.raw()].free_group_count >= kLeafBlockGroups &&
                has_two_free_groups(pages_[page_index.raw()])) {
                return allocate_leaf_block_in_page(page_index);
            }
        }

        pages_.push_back(Page{});
        validate_total_cell_capacity();
        return allocate_leaf_block_in_page(PageIndex{static_cast<uint32_t>(pages_.size() - 1)});
    }

    [[nodiscard]] NodeIndex ensure_node_group(NodeIndex node_index) {
        const NodeCell node = get_node_cell(node_index);
        if (node.child_mask() != 0) {
            return node_index;
        }

        const PageIndex source_page_index{node_index.raw() / cells_per_page()};
        if (pages_[source_page_index.raw()].free_group_count != 0) {
            const CellIndex group_start = allocate_group_in_page(source_page_index);
            NodeCell updated = node;
            updated.set_child_group_index(
                group_index_from_cell_offset(split_index(group_start).second));
            set_node_cell(node_index, updated);
            initialize_node_group(node_index, updated);
            return node_index;
        }

        const CellIndex group_start = find_relocation_node_group();
        NodeCell relocated = node;
        relocated.set_child_group_index(
            group_index_from_cell_offset(split_index(group_start).second));
        initialize_relocated_node_group(group_start, relocated);
        set_forwarding_cell(cell_index_from_node(node_index),
                            cell_index_from_node(relocated_node_index(group_start)));
        return relocated_node_index(group_start);
    }

    [[nodiscard]] NodeIndex ensure_leaf_block(NodeIndex node_index) {
        const NodeCell node = get_node_cell(node_index);
        if (node.child_mask() != 0) {
            return node_index;
        }

        const PageIndex source_page_index{node_index.raw() / cells_per_page()};
        if (pages_[source_page_index.raw()].free_group_count >= kLeafBlockGroups &&
            has_two_free_groups(pages_[source_page_index.raw()])) {
            const CellIndex block_start = allocate_leaf_block_in_page(source_page_index);
            NodeCell updated = node;
            updated.set_child_group_index(
                group_index_from_cell_offset(split_index(block_start).second));
            set_node_cell(node_index, updated);
            initialize_leaf_block(block_start, updated, false);
            return node_index;
        }

        const CellIndex block_start = find_relocation_leaf_block();
        NodeCell relocated = node;
        relocated.set_child_group_index(
            group_index_from_cell_offset(split_index(block_start).second));
        initialize_leaf_block(block_start, relocated, true);
        set_forwarding_cell(cell_index_from_node(node_index), block_start);
        return relocated_leaf_parent_index(block_start);
    }

    void initialize_node_group(NodeIndex node_index, const NodeCell& node) {
        const CellIndex group_start = get_block_start_index(node_index, node);
        for (uint32_t child_index = 0; child_index < kChildCount; ++child_index) {
            init_node_cell(
                node_index_from_cell(CellIndex{group_start.raw() + child_index}),
                make_uniform_node(is_filled(node, child_index)));
        }
        init_node_cell(relocated_node_index(group_start), NodeCell{});
    }

    void initialize_relocated_node_group(CellIndex group_start, const NodeCell& node) {
        for (uint32_t child_index = 0; child_index < kChildCount; ++child_index) {
            init_node_cell(
                node_index_from_cell(CellIndex{group_start.raw() + child_index}),
                make_uniform_node(is_filled(node, child_index)));
        }
        init_node_cell(relocated_node_index(group_start), node);
    }

    void initialize_leaf_block(CellIndex block_start,
                               const NodeCell& node,
                               bool relocated) {
        init_node_cell(relocated_leaf_parent_index(block_start),
                       relocated ? node : NodeCell{});

        for (uint32_t child_index = 0; child_index < kChildCount; ++child_index) {
            const uint64_t initial = is_filled(node, child_index) ? kLeafFullMask : 0;
            init_leaf_value(CellIndex{block_start.raw() + kLeafDataStartSlot +
                                      child_index * kLeafWordsPerLeaf},
                            initial);
        }

        initialize_cell_value(CellIndex{block_start.raw() + kLeafUnusedSlot}, 0);
    }

    void free_node_group(CellIndex group_start_absolute_index) {
        const auto [page_index, cell_offset] = split_index(group_start_absolute_index);
        if ((cell_offset.raw() % kGroupSize) != 0) {
            throw std::logic_error(
                "SVOForwardingPacked: group free requires group-aligned index");
        }

        Page& page = pages_[page_index.raw()];
        for (uint32_t i = 0; i < kNodeGroupCells; ++i) {
            update_cell_value(CellIndex{group_start_absolute_index.raw() + i}, 0);
        }

        release_group(page, group_index_from_cell_offset(cell_offset));
        --used_group_count_;
    }

    void free_leaf_block(CellIndex block_start_absolute_index) {
        const auto [page_index, cell_offset] = split_index(block_start_absolute_index);
        if ((cell_offset.raw() % kGroupSize) != 0) {
            throw std::logic_error(
                "SVOForwardingPacked: leaf free requires group-aligned index");
        }

        Page& page = pages_[page_index.raw()];
        for (uint32_t i = 0; i < kLeafBlockCells; ++i) {
            update_cell_value(CellIndex{block_start_absolute_index.raw() + i}, 0);
        }

        const GroupIndex first_group = group_index_from_cell_offset(cell_offset);
        release_group(page, first_group);
        release_group(page, GroupIndex{first_group.raw() + 1});
        used_group_count_ -= kLeafBlockGroups;
    }

    void free_target_storage(CellIndex target, uint32_t depth) {
        if (depth == max_depth_) {
            free_leaf_block(target);
        } else {
            free_node_group(align_to_group_start(target));
        }
    }

    void free_forward_target(CellIndex head_index, uint32_t depth) {
        const uint32_t head_value = get_cell_value(head_index);
        if (!is_forwarding_cell_value(head_value)) {
            return;
        }

        free_target_storage(CellIndex{head_value & kForwardingIndexMask}, depth);
    }

    void collapse_path(const std::vector<NodeIndex>& node_indices,
                       const std::vector<bool>& node_relocated,
                       const std::vector<uint32_t>& child_indices) {
        for (std::size_t i = node_indices.size(); i-- > 0;) {
            const uint32_t depth = static_cast<uint32_t>(i);
            const NodeIndex node_index = node_indices[i];
            const NodeCell node = get_node_cell(node_index);
            if (node.child_mask() != 0) {
                break;
            }
            if (node.filled_mask() != 0u && node.filled_mask() != 0xFFu) {
                if (i != 0) {
                    break;
                }
            }
            if (i == 0) {
                NodeCell root = node;
                root.set_child_group_index(GroupIndex{0});
                if (node_relocated[0]) {
                    const uint32_t head_value = get_cell_value(root_cell_index());
                    const CellIndex target{head_value & kForwardingIndexMask};
                    set_node_cell(root_node_index(), root);
                    free_target_storage(target, depth);
                } else if (node.child_group_index().raw() != 0) {
                    if (depth == max_depth_) {
                        free_leaf_block(get_block_start_index(node_index, node));
                    } else {
                        free_node_group(get_block_start_index(node_index, node));
                    }
                    set_node_cell(root_node_index(), root);
                }
                break;
            }

            const bool filled = node.filled_mask() == 0xFFu;
            const NodeIndex parent_index = node_indices[i - 1];
            NodeCell parent = get_node_cell(parent_index);
            const uint32_t child_index = child_indices[i - 1];
            const CellIndex child_cell =
                get_child_node_cell_index(parent_index, parent, child_index);

            set_child_mask(parent, child_index, false);
            set_filled(parent, child_index, filled);
            free_forward_target(child_cell, depth);
            if (parent.child_mask() == 0) {
                if (!node_relocated[i - 1]) {
                    if ((i - 1) == max_depth_) {
                        free_leaf_block(get_block_start_index(parent_index, parent));
                    } else {
                        free_node_group(get_block_start_index(parent_index, parent));
                    }
                }
                parent.set_child_group_index(GroupIndex{0});
            }
            set_node_cell(parent_index, parent);
        }
    }

    [[nodiscard]] uint32_t get_child_index(uint32_t x,
                                           uint32_t y,
                                           uint32_t z,
                                           uint32_t depth) const {
        const uint32_t scale = LEAF_VOXEL_COUNT << (max_depth_ - depth);
        const uint32_t x_bit = (x / scale) & 1u;
        const uint32_t y_bit = (y / scale) & 1u;
        const uint32_t z_bit = (z / scale) & 1u;
        return x_bit | (y_bit << 1) | (z_bit << 2);
    }

    [[nodiscard]] static uint32_t get_leaf_offset(uint32_t x,
                                                  uint32_t y,
                                                  uint32_t z) {
        const uint32_t x_offset = x & (LEAF_VOXEL_COUNT - 1u);
        const uint32_t y_offset = y & (LEAF_VOXEL_COUNT - 1u);
        const uint32_t z_offset = z & (LEAF_VOXEL_COUNT - 1u);
        return x_offset + y_offset * LEAF_VOXEL_COUNT +
               z_offset * LEAF_VOXEL_COUNT * LEAF_VOXEL_COUNT;
    }

    void validate_coordinates(uint32_t x, uint32_t y, uint32_t z) const {
        if (x >= world_voxel_count() || y >= world_voxel_count() ||
            z >= world_voxel_count()) {
            throw std::out_of_range(
                "SVOForwardingPacked: voxel coordinates out of bounds");
        }
    }

    void validate_max_depth() const {
        if (max_depth_ > kMaxSupportedDepth) {
            throw std::out_of_range(
                "SVOForwardingPacked: max_depth out of supported range");
        }
    }

    [[nodiscard]] std::pair<PageIndex, CellOffset> split_index(
        CellIndex absolute_index) const {
        const PageIndex page_index{absolute_index.raw() / cells_per_page()};
        const CellOffset cell_offset{absolute_index.raw() % cells_per_page()};
        if (page_index.raw() >= pages_.size()) {
            throw std::out_of_range(
                "SVOForwardingPacked: absolute index out of bounds");
        }
        return {page_index, cell_offset};
    }

    [[nodiscard]] CellIndex align_to_group_start(CellIndex absolute_index) const {
        const uint32_t page_base =
            (absolute_index.raw() / cells_per_page()) * cells_per_page();
        const uint32_t page_offset = absolute_index.raw() % cells_per_page();
        return CellIndex{page_base + page_offset - (page_offset % kGroupSize)};
    }

    void validate_total_cell_capacity() const {
        if (pages_.size() * std::size_t(cells_per_page()) >
            std::size_t(kForwardingIndexMask) + 1) {
            throw std::overflow_error("SVOForwardingPacked: absolute index overflow");
        }
    }

    [[nodiscard]] static bool is_group_free(const Page& page, GroupIndex group_index) {
        const uint32_t word_index = group_index.raw() / 64u;
        const uint32_t bit_index = group_index.raw() % 64u;
        return (page.free_group_bitmap[word_index] & (uint64_t{1} << bit_index)) != 0;
    }

    static void mark_group_used(Page& page, GroupIndex group_index) {
        const uint32_t word_index = group_index.raw() / 64u;
        const uint32_t bit_index = group_index.raw() % 64u;
        const uint64_t mask = uint64_t{1} << bit_index;
        if ((page.free_group_bitmap[word_index] & mask) == 0) {
            throw std::logic_error("SVOForwardingPacked: allocating used child group");
        }
        page.free_group_bitmap[word_index] &= ~mask;
        --page.free_group_count;
    }

    [[nodiscard]] static GroupIndex take_free_group(Page& page) {
        for (uint32_t group = 0; group < kGroupsPerPage; ++group) {
            const GroupIndex index{group};
            if (!is_group_free(page, index)) {
                continue;
            }
            mark_group_used(page, index);
            return index;
        }
        throw std::overflow_error(
            "SVOForwardingPacked: page has no free child group");
    }

    [[nodiscard]] static bool has_two_free_groups(const Page& page) {
        for (uint32_t group = 0; group + 1 < kGroupsPerPage; ++group) {
            if (is_group_free(page, GroupIndex{group}) &&
                is_group_free(page, GroupIndex{group + 1})) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] static GroupIndex take_two_free_groups(Page& page) {
        for (uint32_t group = 0; group + 1 < kGroupsPerPage; ++group) {
            const GroupIndex first{group};
            const GroupIndex second{group + 1};
            if (!is_group_free(page, first) || !is_group_free(page, second)) {
                continue;
            }
            mark_group_used(page, first);
            mark_group_used(page, second);
            return first;
        }
        throw std::overflow_error(
            "SVOForwardingPacked: page has no free contiguous leaf block");
    }

    static void release_group(Page& page, GroupIndex group_index) {
        const uint32_t word_index = group_index.raw() / 64u;
        const uint32_t bit_index = group_index.raw() % 64u;
        const uint64_t mask = uint64_t{1} << bit_index;
        if ((page.free_group_bitmap[word_index] & mask) != 0) {
            throw std::logic_error("SVOForwardingPacked: double free child group");
        }
        page.free_group_bitmap[word_index] |= mask;
        ++page.free_group_count;
    }

    std::vector<Page> pages_;
    std::size_t used_group_count_ = 0;
    std::size_t forwarding_cell_count_ = 0;
    uint32_t max_depth_;
};

using SVOForwardingPackedLeaf = SVOForwardingPackedLeafImpl<>;

} // namespace algo::svt
