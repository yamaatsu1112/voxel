#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace algo::memory {

template <std::size_t PageWords, typename Word = uint32_t>
class VirtualWordMemory {
    static_assert(PageWords > 0);
    static_assert(std::has_single_bit(PageWords));

  public:
    using word_type = Word;

    static constexpr uint32_t kInvalidPage =
        std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kVirtualPageCount =
        uint32_t{1} << (32 - std::countr_zero(PageWords));

    struct Page {
        std::array<Word, PageWords> words = {};
    };

    [[nodiscard]] Word read(uint32_t address) const {
        const uint32_t page_id = address / PageWords;
        const uint32_t physical_page = page_table_[page_id];
        if (physical_page == kInvalidPage) {
            return Word{};
        }
        return pages_[physical_page].words[address % PageWords];
    }

    void write(uint32_t address, Word value) {
        const uint32_t page_id = address / PageWords;
        uint32_t& physical_page = page_table_[page_id];
        if (physical_page == kInvalidPage) {
            physical_page = allocate_page();
        }
        pages_[physical_page].words[address % PageWords] = value;
    }

    [[nodiscard]] std::size_t mapped_page_count() const {
        return pages_.size() - free_pages_.size();
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const {
        return page_table_.capacity() * sizeof(uint32_t) +
               pages_.capacity() * sizeof(Page) +
               free_pages_.capacity() * sizeof(uint32_t);
    }

  private:
    [[nodiscard]] uint32_t allocate_page() {
        if (!free_pages_.empty()) {
            const uint32_t page = free_pages_.back();
            free_pages_.pop_back();
            pages_[page] = Page{};
            return page;
        }

        const uint32_t page = static_cast<uint32_t>(pages_.size());
        pages_.push_back(Page{});
        return page;
    }

    std::vector<uint32_t> page_table_ =
        std::vector<uint32_t>(kVirtualPageCount, kInvalidPage);
    std::vector<Page> pages_;
    std::vector<uint32_t> free_pages_;
};

} // namespace algo::memory
