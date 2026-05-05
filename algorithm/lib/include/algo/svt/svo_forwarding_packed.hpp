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

inline constexpr uint32_t kDefaultPackedForwardingPageBits = 15;

template <uint32_t PageBits = kDefaultPackedForwardingPageBits>
class SVOForwardingPackedImpl {
  public:
    explicit SVOForwardingPackedImpl(uint32_t max_depth = DEFAULT_MAX_DEPTH)
        : max_depth_(max_depth) {
        validate_max_depth();
        pages_.push_back(Page{});
        mark_group_used(pages_[0], GroupIndex{0});
        init_node_cell(root_node_index(), NodeCell{});
    }

    void set_voxel(uint32_t x, uint32_t y, uint32_t z, bool value) {
        validate_coordinates(x, y, z);

        auto [node_index, node] = resolve_node(root_cell_index());
        std::vector<NodeIndex> node_indices;
        node_indices.reserve(max_depth_ + 1);
        std::vector<uint32_t> child_indices;
        child_indices.reserve(max_depth_);
        node_indices.push_back(node_index);

        for (uint32_t depth = 0; depth < max_depth_; ++depth) {
            const uint32_t child_index = get_child_index(x, y, z, depth);
            child_indices.push_back(child_index);

            if (has_child(node, child_index)) {
                auto [next_index, next_node] = resolve_node(
                    get_child_cell_index(node_index, node, child_index));
                node_index = next_index;
                node = next_node;
                node_indices.push_back(node_index);
                continue;
            }

            const bool filled = is_filled(node, child_index);
            if (filled == value) {
                return;
            }

            node_index = ensure_child_group(node_index);
            node_indices.back() = node_index;

            node = get_node_cell(node_index);
            set_child_mask(node, child_index, true);
            set_node_cell(node_index, node);

            const NodeIndex child_cell_index =
                node_index_from_cell(get_child_cell_index(node_index, node, child_index));
            set_node_cell(child_cell_index, make_uniform_node(filled));
            node_index = child_cell_index;
            node = get_node_cell(node_index);
            node_indices.push_back(node_index);
        }

        NodeCell terminal = get_node_cell(node_index);
        const uint32_t bit_offset = get_terminal_offset(x, y, z);
        if (is_filled(terminal, bit_offset) == value) {
            return;
        }
        set_filled(terminal, bit_offset, value);
        set_node_cell(node_index, terminal);

        collapse_path(node_indices, child_indices);
    }

    [[nodiscard]] bool get_voxel(uint32_t x, uint32_t y, uint32_t z) const {
        validate_coordinates(x, y, z);

        auto [node_index, node] = resolve_node(root_cell_index());
        for (uint32_t depth = 0; depth < max_depth_; ++depth) {
            const uint32_t child_index = get_child_index(x, y, z, depth);
            if (!has_child(node, child_index)) {
                return is_filled(node, child_index);
            }
            auto [next_index, next_node] = resolve_node(
                get_child_cell_index(node_index, node, child_index));
            node_index = next_index;
            node = next_node;
        }

        return is_filled(node, get_terminal_offset(x, y, z));
    }

    [[nodiscard]] std::size_t node_count() const {
        return node_cell_count_;
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
        return kTerminalVoxelCount << max_depth;
    }

    [[nodiscard]] static constexpr uint32_t
    max_depth_for_world_size(uint32_t requested_world_size) {
        return detail::depth_for_world_size(
            requested_world_size, kTerminalVoxelCount, 1, 1,
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
    static constexpr uint32_t kRelocatedNodeSlot = kChildCount;
    static constexpr uint32_t kRootNodeIndex = 0;
    static constexpr uint32_t kNodeTagBit = 0u;
    static constexpr uint32_t kForwardingTagBit = 1u << 31;
    static constexpr uint32_t kChildMaskMask = 0x000000FFu;
    static constexpr uint32_t kFilledMaskMask = 0x0000FF00u;
    static constexpr uint32_t kLocalIndexMask = 0x7FFF0000u;
    static constexpr uint32_t kFilledShift = 8;
    static constexpr uint32_t kLocalIndexShift = 16;
    static constexpr uint32_t kForwardingIndexMask = 0x7FFFFFFFu;
    static constexpr uint32_t kTerminalVoxelCount = 2;
    static constexpr uint32_t kMaxSupportedDepth = 29;
    static constexpr uint32_t kPageBits = PageBits;
    static constexpr uint32_t kGroupsPerPage = (1u << kPageBits) / kGroupSize;
    static constexpr uint32_t kFreeGroupBitmapWords =
        (kGroupsPerPage + 63u) / 64u;
    static_assert(kPageBits >= 7 && kPageBits <= kDefaultPackedForwardingPageBits,
                  "PageBits out of supported range");
    static_assert(kGroupsPerPage > 0, "Page must have room for at least one group");

    struct PageIndex final {
      public:
        explicit constexpr PageIndex(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct CellOffset final {
      public:
        explicit constexpr CellOffset(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct GroupIndex final {
      public:
        explicit constexpr GroupIndex(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct CellIndex final {
      public:
        explicit constexpr CellIndex(uint32_t value) : value_(value) {}
        [[nodiscard]] constexpr uint32_t raw() const { return value_; }

      private:
        uint32_t value_;
    };

    struct NodeIndex final {
      public:
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

    [[nodiscard]] uint32_t cells_per_page() const {
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

    [[nodiscard]] static uint32_t pack_forwarding(NodeIndex target_index) {
        return kForwardingTagBit | (target_index.raw() & kForwardingIndexMask);
    }

    [[nodiscard]] static bool is_forwarding_cell_value(uint32_t cell) {
        return (cell & kForwardingTagBit) != kNodeTagBit;
    }

    [[nodiscard]] uint32_t get_cell_value(CellIndex absolute_index) const {
        const auto [page_index, cell_offset] = split_index(absolute_index);
        return pages_[page_index.raw()].cells[cell_offset.raw()];
    }

    [[nodiscard]] NodeCell get_node_cell(NodeIndex node_index) const {
        return NodeCell{get_cell_value(cell_index_from_node(node_index))};
    }

    void initialize_cell_value(CellIndex absolute_index, uint32_t value) {
        const auto [page_index, cell_offset] = split_index(absolute_index);
        pages_[page_index.raw()].cells[cell_offset.raw()] = value;
        if (is_forwarding_cell_value(value)) {
            ++forwarding_cell_count_;
        } else {
            ++node_cell_count_;
        }
    }

    void update_cell_value(CellIndex absolute_index, uint32_t value) {
        const auto [page_index, cell_offset] = split_index(absolute_index);
        Page& page = pages_[page_index.raw()];
        if (page.cells[cell_offset.raw()] == value) {
            return;
        }
        if (is_forwarding_cell_value(page.cells[cell_offset.raw()])) {
            --forwarding_cell_count_;
        } else {
            --node_cell_count_;
        }
        if (is_forwarding_cell_value(value)) {
            ++forwarding_cell_count_;
        } else {
            ++node_cell_count_;
        }
        page.cells[cell_offset.raw()] = value;
    }

    void init_node_cell(NodeIndex node_index, const NodeCell& node) {
        initialize_cell_value(cell_index_from_node(node_index), node.value);
    }

    void set_node_cell(NodeIndex node_index, const NodeCell& node) {
        update_cell_value(cell_index_from_node(node_index), node.value);
    }

    void set_forwarding_cell(CellIndex absolute_index, NodeIndex target_index) {
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

    [[nodiscard]] NodeIndex node_index_at(CellIndex group_start, uint32_t slot_offset) const {
        return node_index_from_cell(CellIndex{group_start.raw() + slot_offset});
    }

    [[nodiscard]] NodeIndex relocated_node_index(CellIndex group_start) const {
        return node_index_at(group_start, kRelocatedNodeSlot);
    }

    [[nodiscard]] CellIndex allocate_group_in_page(PageIndex page_index) {
        Page& page = pages_[page_index.raw()];
        if (page.free_group_count == 0) {
            throw std::overflow_error(
                "SVOForwardingPacked: page has no free child group");
        }
        const GroupIndex group_index = take_free_group(page);
        return group_start_index(page_index, group_index);
    }

    [[nodiscard]] CellIndex find_relocation_group() {
        for (uint32_t raw_page_index = 0; raw_page_index < pages_.size();
             ++raw_page_index) {
            const PageIndex page_index{raw_page_index};
            if (pages_[page_index.raw()].free_group_count != 0) {
                return allocate_group_in_page(page_index);
            }
        }

        pages_.push_back(Page{});
        if (pages_.size() * std::size_t(cells_per_page()) >
            std::size_t(kForwardingIndexMask) + 1) {
            throw std::overflow_error("SVOForwardingPacked: absolute index overflow");
        }
        const PageIndex page_index{static_cast<uint32_t>(pages_.size() - 1)};
        return allocate_group_in_page(page_index);
    }

    [[nodiscard]] NodeIndex ensure_child_group(NodeIndex node_index) {
        NodeCell node = get_node_cell(node_index);
        if (node.child_mask() != 0) {
            return node_index;
        }

        const PageIndex source_page_index{node_index.raw() / cells_per_page()};
        if (pages_[source_page_index.raw()].free_group_count != 0) {
            const CellIndex group_start = allocate_group_in_page(source_page_index);
            node.set_child_group_index(
                group_index_from_cell_offset(split_index(group_start).second));
            set_node_cell(node_index, node);
            initialize_child_group(node_index, node);
            return node_index;
        }

        const CellIndex group_start = find_relocation_group();
        node.set_child_group_index(
            group_index_from_cell_offset(split_index(group_start).second));
        const NodeIndex relocated_index = relocated_node_index(group_start);
        initialize_relocated_group(group_start, node);
        set_forwarding_cell(cell_index_from_node(node_index), relocated_index);
        return relocated_index;
    }

    void initialize_child_group(NodeIndex node_index, const NodeCell& node) {
        const CellIndex group_start = get_group_start_index(node_index, node);
        for (uint32_t child_index = 0; child_index < kChildCount; ++child_index) {
            init_node_cell(node_index_at(group_start, child_index),
                           make_uniform_node(is_filled(node, child_index)));
        }
        init_node_cell(relocated_node_index(group_start), NodeCell{});
    }

    void initialize_relocated_group(CellIndex group_start, const NodeCell& node) {
        for (uint32_t child_index = 0; child_index < kChildCount; ++child_index) {
            init_node_cell(node_index_at(group_start, child_index),
                           make_uniform_node(is_filled(node, child_index)));
        }
        init_node_cell(relocated_node_index(group_start), node);
    }

    [[nodiscard]] CellIndex get_group_start_index(NodeIndex node_index,
                                                  const NodeCell& node) const {
        const PageIndex page_index{node_index.raw() / cells_per_page()};
        return group_start_index(page_index, node.child_group_index());
    }

    [[nodiscard]] bool is_relocated_node(NodeIndex node_index) const {
        const uint32_t page_offset = node_index.raw() % cells_per_page();
        return (page_offset % kGroupSize) == kRelocatedNodeSlot;
    }

    [[nodiscard]] CellIndex get_child_cell_index(NodeIndex node_index,
                                                 const NodeCell& node,
                                                 uint32_t child_index) const {
        return CellIndex{get_group_start_index(node_index, node).raw() + child_index};
    }

    void free_group(CellIndex group_start_absolute_index) {
        const auto [page_index, cell_offset] = split_index(group_start_absolute_index);
        Page& page = pages_[page_index.raw()];

        if ((cell_offset.raw() % kGroupSize) != 0) {
            throw std::logic_error(
                "SVOForwardingPacked: group free requires group-aligned index");
        }

        for (uint32_t i = 0; i < kGroupSize; ++i) {
            const uint32_t idx = cell_offset.raw() + i;
            if (is_forwarding_cell_value(page.cells[idx])) {
                --forwarding_cell_count_;
            } else {
                --node_cell_count_;
            }
            page.cells[idx] = 0;
        }

        const GroupIndex group_index = group_index_from_cell_offset(cell_offset);
        release_group(page, group_index);
    }

    void free_forward_target(CellIndex head_index) {
        const uint32_t head_value = get_cell_value(head_index);
        if (!is_forwarding_cell_value(head_value)) {
            return;
        }
        const CellIndex target{head_value & kForwardingIndexMask};
        free_group(align_to_group_start(target));
    }

    void collapse_path(const std::vector<NodeIndex>& node_indices,
                       const std::vector<uint32_t>& child_indices) {
        for (std::size_t i = node_indices.size(); i-- > 1;) {
            const NodeIndex node_index = node_indices[i];
            const NodeCell node = get_node_cell(node_index);
            if (node.child_mask() != 0) {
                break;
            }
            if (node.filled_mask() != 0u && node.filled_mask() != 0xFFu) {
                break;
            }

            const bool filled = node.filled_mask() == 0xFFu;
            const NodeIndex parent_index = node_indices[i - 1];
            NodeCell parent = get_node_cell(parent_index);
            const uint32_t child_index = child_indices[i - 1];
            const CellIndex child_cell_index =
                get_child_cell_index(parent_index, parent, child_index);

            set_child_mask(parent, child_index, false);
            set_filled(parent, child_index, filled);
            free_forward_target(child_cell_index);
            if (parent.child_mask() == 0) {
                if (!is_relocated_node(parent_index)) {
                    free_group(get_group_start_index(parent_index, parent));
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
        const uint32_t scale = kTerminalVoxelCount << (max_depth_ - depth - 1);
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
                "SVOForwardingPacked: voxel coordinates out of bounds");
        }
    }

    void validate_max_depth() const {
        if (max_depth_ == 0 || max_depth_ > kMaxSupportedDepth) {
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

    static void mark_group_used(Page& page, GroupIndex group_index) {
        const uint32_t word_index = group_index.raw() / 64u;
        const uint32_t bit_index = group_index.raw() % 64u;
        const uint64_t mask = uint64_t{1} << bit_index;
        if ((page.free_group_bitmap[word_index] & mask) == 0) {
            throw std::logic_error(
                "SVOForwardingPacked: allocating used child group");
        }
        page.free_group_bitmap[word_index] &= ~mask;
        --page.free_group_count;
    }

    [[nodiscard]] static GroupIndex take_free_group(Page& page) {
        for (uint32_t word_index = 0; word_index < page.free_group_bitmap.size();
             ++word_index) {
            const uint64_t word = page.free_group_bitmap[word_index];
            if (word == 0) {
                continue;
            }
            const uint32_t bit_index = std::countr_zero(word);
            const GroupIndex group_index{word_index * 64u + bit_index};
            const uint64_t mask = uint64_t{1} << bit_index;
            page.free_group_bitmap[word_index] &= ~mask;
            --page.free_group_count;
            return group_index;
        }
        throw std::overflow_error(
            "SVOForwardingPacked: page has no free child group");
    }

    static void release_group(Page& page, GroupIndex group_index) {
        const uint32_t word_index = group_index.raw() / 64u;
        const uint32_t bit_index = group_index.raw() % 64u;
        const uint64_t mask = uint64_t{1} << bit_index;
        if ((page.free_group_bitmap[word_index] & mask) != 0) {
            throw std::logic_error(
                "SVOForwardingPacked: double free child group");
        }
        page.free_group_bitmap[word_index] |= mask;
        ++page.free_group_count;
    }

    std::vector<Page> pages_;
    std::size_t node_cell_count_ = 0;
    std::size_t forwarding_cell_count_ = 0;
    uint32_t max_depth_;
};

using SVOForwardingPacked = SVOForwardingPackedImpl<>;

} // namespace algo::svt
