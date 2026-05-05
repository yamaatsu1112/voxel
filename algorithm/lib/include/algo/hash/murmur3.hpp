#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace algo::hash {

class Murmur3x86_32 {
  public:
    explicit Murmur3x86_32(uint32_t seed = 0) : hash_(seed) {}

    void mix_word(uint32_t word) {
        word *= 0xcc9e2d51u;
        word = std::rotl(word, 15);
        word *= 0x1b873593u;

        hash_ ^= word;
        hash_ = std::rotl(hash_, 13);
        hash_ = hash_ * 5u + 0xe6546b64u;
    }

    [[nodiscard]] uint32_t finish(uint32_t byte_count) const {
        uint32_t hash = hash_;
        hash ^= byte_count;
        hash ^= hash >> 16;
        hash *= 0x85ebca6bu;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35u;
        hash ^= hash >> 16;
        return hash;
    }

  private:
    uint32_t hash_;
};

[[nodiscard]] inline uint32_t murmur3_words(std::span<const uint32_t> words,
                                            uint32_t seed = 0) {
    Murmur3x86_32 hash(seed);
    for (const uint32_t word : words) {
        hash.mix_word(word);
    }
    return hash.finish(static_cast<uint32_t>(words.size() * sizeof(uint32_t)));
}

[[nodiscard]] inline uint32_t murmur3_u64(uint64_t value, uint32_t seed = 0) {
    const uint32_t words[] = {
        static_cast<uint32_t>(value),
        static_cast<uint32_t>(value >> 32),
    };
    return murmur3_words(words, seed);
}

} // namespace algo::hash
