#pragma once

/**
 * @file atomic_bit_vector.hpp
 * @brief Compact packed Boolean vectors for HighVoronoi state and work flags.
 *
 * The file deliberately provides two classes with the same bit-oriented API:
 *
 * - `BitVector` stores ordinary `std::uint64_t` words and is intended for
 *   structural state that is not mutated concurrently, such as HighVoronoi's
 *   `active` and `visible` flags.
 * - `AtomicBitVector` stores `std::atomic<std::uint64_t>` words and is intended
 *   for flags collected by several ComputeVoronoi workers, such as `affected`
 *   nodes and periodic mirror requests.
 *
 * Both classes pack 64 Boolean values per word. Individual out-of-range writes
 * are ignored and out-of-range reads return false. `clear()` resets all bits
 * while preserving the logical size. Structural operations on an
 * `AtomicBitVector` are permitted only between parallel compute phases. In
 * particular, `resize()` allocates fresh atomic storage and preserves the old
 * prefix; it must never run concurrently with bit updates.
 */

#include <atomic>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace highvoronoi::detail {

/**
 * @brief Packed non-atomic Boolean vector.
 *
 * This class is the inexpensive structural counterpart of AtomicBitVector.
 * It is not thread-safe for concurrent mutation. Concurrent read-only access is
 * safe when no thread changes its size or stored words.
 */
class BitVector final {
public:
    using Word = std::uint64_t;

    static constexpr std::size_t bits_per_word =
        std::numeric_limits<Word>::digits;

    BitVector() = default;

    explicit BitVector(std::size_t bit_count, bool value = false)
        : bit_count_(bit_count),
          words_(word_count_for(bit_count), value ? ~Word{0} : Word{0}) {
        clear_unused_tail_bits();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return bit_count_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return bit_count_ == 0;
    }

    /** Set one bit. Out-of-range indices are ignored. */
    void set(std::size_t index) noexcept {
        if (index >= bit_count_) {
            return;
        }
        words_[word_index(index)] |= bit_mask(index);
    }

    /** Clear one bit. Out-of-range indices are ignored. */
    void reset(std::size_t index) noexcept {
        if (index >= bit_count_) {
            return;
        }
        words_[word_index(index)] &= ~bit_mask(index);
    }

    /** Toggle one bit. Out-of-range indices are ignored. */
    void flip(std::size_t index) noexcept {
        if (index >= bit_count_) {
            return;
        }
        words_[word_index(index)] ^= bit_mask(index);
    }

    /** Return one bit. Out-of-range indices read as false. */
    [[nodiscard]] bool test(std::size_t index) const noexcept {
        if (index >= bit_count_) {
            return false;
        }
        return (words_[word_index(index)] & bit_mask(index)) != Word{0};
    }

    /** Clear every stored bit without changing the logical size. */
    void clear() noexcept {
        std::fill(words_.begin(), words_.end(), Word{0});
    }

    /** Set every logical bit without changing the logical size. */
    void set_all() noexcept {
        std::fill(words_.begin(), words_.end(), ~Word{0});
        clear_unused_tail_bits();
    }

    /**
     * @brief Resize while preserving the old prefix.
     *
     * Newly appended bits receive `value`. Shrinking discards the removed tail.
     */
    void resize(std::size_t new_bit_count, bool value = false) {
        const std::size_t old_bit_count = bit_count_;
        if (new_bit_count == old_bit_count) {
            return;
        }

        const std::size_t new_word_count = word_count_for(new_bit_count);
        words_.resize(new_word_count, Word{0});
        bit_count_ = new_bit_count;

        if (new_bit_count > old_bit_count && value) {
            for (std::size_t index = old_bit_count;
                 index < new_bit_count;
                 ++index) {
                set(index);
            }
        }

        clear_unused_tail_bits();
    }

    /** Append one Boolean flag. */
    void push_back(bool value) {
        const std::size_t index = bit_count_;
        resize(bit_count_ + std::size_t{1}, false);
        if (value) {
            set(index);
        }
    }

    /** Reserve enough packed words for at least `bit_capacity` bits. */
    void reserve(std::size_t bit_capacity) {
        words_.reserve(word_count_for(bit_capacity));
    }

private:
    [[nodiscard]] static constexpr std::size_t
    word_count_for(std::size_t bit_count) noexcept {
        return bit_count / bits_per_word +
               (bit_count % bits_per_word != 0 ? std::size_t{1}
                                                : std::size_t{0});
    }

    [[nodiscard]] static constexpr std::size_t
    word_index(std::size_t index) noexcept {
        return index / bits_per_word;
    }

    [[nodiscard]] static constexpr Word
    bit_mask(std::size_t index) noexcept {
        return Word{1} << (index % bits_per_word);
    }

    void clear_unused_tail_bits() noexcept {
        if (words_.empty() || bit_count_ == 0) {
            return;
        }
        const std::size_t used = bit_count_ % bits_per_word;
        if (used == 0) {
            return;
        }
        const Word mask = (Word{1} << used) - Word{1};
        words_.back() &= mask;
    }

    std::size_t bit_count_ = 0;
    std::vector<Word> words_;
};

/**
 * @brief Compact thread-safe bit flags backed by atomic 64-bit words.
 *
 * AtomicBitVector stores 64 Boolean flags in each std::atomic<std::uint64_t>.
 * Individual bits may be set, reset, flipped, and tested concurrently without
 * losing updates to other bits in the same word. All operations use relaxed
 * memory ordering because the class provides atomic flag storage, not
 * inter-thread publication or phase synchronization.
 *
 * Concurrent set/reset/flip operations are data-race free. Calling clear()
 * concurrently with bit updates is also data-race free at the C++ memory-model
 * level, but the resulting logical state is intentionally unspecified; callers
 * should clear the vector only between computation phases.
 */
class AtomicBitVector final {
public:
    using Word = std::uint64_t;
    using AtomicWord = std::atomic<Word>;

    static constexpr std::size_t bits_per_word =
        std::numeric_limits<Word>::digits;

    explicit AtomicBitVector(std::size_t bit_count)
        : bit_count_(bit_count),
          word_count_(
              bit_count / bits_per_word +
              (bit_count % bits_per_word != 0 ? std::size_t{1}
                                               : std::size_t{0})),
          words_(
              word_count_ == 0
                  ? nullptr
                  : std::make_unique<AtomicWord[]>(word_count_)) {
        // In C++17 a default-constructed std::atomic<T> does not initialize
        // its contained T. atomic_init establishes the initial zero value.
        for (std::size_t i = 0; i < word_count_; ++i) {
            std::atomic_init(&words_[i], Word{0});
        }
    }

    AtomicBitVector(const AtomicBitVector&) = delete;
    AtomicBitVector& operator=(const AtomicBitVector&) = delete;
    AtomicBitVector(AtomicBitVector&&) noexcept = default;
    AtomicBitVector& operator=(AtomicBitVector&&) noexcept = default;

    [[nodiscard]] std::size_t size() const noexcept {
        return bit_count_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return bit_count_ == 0;
    }

    /** Set one bit. Out-of-range indices are ignored. */
    void set(std::size_t index) noexcept {
        if (index >= bit_count_) {
            return;
        }

        words_[word_index(index)].fetch_or(
            bit_mask(index),
            std::memory_order_relaxed);
    }

    /** Clear one bit. Out-of-range indices are ignored. */
    void reset(std::size_t index) noexcept {
        if (index >= bit_count_) {
            return;
        }

        words_[word_index(index)].fetch_and(
            ~bit_mask(index),
            std::memory_order_relaxed);
    }

    /** Toggle one bit. Out-of-range indices are ignored. */
    void flip(std::size_t index) noexcept {
        if (index >= bit_count_) {
            return;
        }

        words_[word_index(index)].fetch_xor(
            bit_mask(index),
            std::memory_order_relaxed);
    }

    /** Return one bit. Out-of-range indices read as false. */
    [[nodiscard]] bool test(std::size_t index) const noexcept {
        if (index >= bit_count_) {
            return false;
        }

        return (
            words_[word_index(index)].load(std::memory_order_relaxed) &
            bit_mask(index)) != Word{0};
    }

    /** Clear every stored bit. Call between logical computation phases. */
    void clear() noexcept {
        for (std::size_t i = 0; i < word_count_; ++i) {
            words_[i].store(Word{0}, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Resize while preserving the old prefix.
     *
     * This is a structural operation for use strictly between parallel
     * ComputeVoronoi phases. No thread may access this vector while resize()
     * runs. Fresh atomic storage is allocated because std::atomic words are not
     * copyable or movable as individual vector elements.
     */
    void resize(std::size_t new_bit_count, bool value = false) {
        if (new_bit_count == bit_count_) {
            return;
        }

        const std::size_t old_bit_count = bit_count_;
        const std::size_t new_word_count =
            new_bit_count / bits_per_word +
            (new_bit_count % bits_per_word != 0 ? std::size_t{1}
                                                 : std::size_t{0});

        std::unique_ptr<AtomicWord[]> new_words =
            new_word_count == 0
                ? nullptr
                : std::make_unique<AtomicWord[]>(new_word_count);

        for (std::size_t i = 0; i < new_word_count; ++i) {
            std::atomic_init(&new_words[i], Word{0});
        }

        const std::size_t copied_words = std::min(word_count_, new_word_count);
        for (std::size_t i = 0; i < copied_words; ++i) {
            new_words[i].store(
                words_[i].load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        }

        words_ = std::move(new_words);
        bit_count_ = new_bit_count;
        word_count_ = new_word_count;

        if (new_bit_count > old_bit_count && value) {
            for (std::size_t index = old_bit_count;
                 index < new_bit_count;
                 ++index) {
                set(index);
            }
        }

        clear_unused_tail_bits();
    }

private:
    [[nodiscard]] static constexpr std::size_t
    word_index(std::size_t index) noexcept {
        return index / bits_per_word;
    }

    [[nodiscard]] static constexpr Word
    bit_mask(std::size_t index) noexcept {
        return Word{1} << (index % bits_per_word);
    }

    void clear_unused_tail_bits() noexcept {
        if (word_count_ == 0 || bit_count_ == 0) {
            return;
        }
        const std::size_t used = bit_count_ % bits_per_word;
        if (used == 0) {
            return;
        }
        const Word mask = (Word{1} << used) - Word{1};
        words_[word_count_ - 1].fetch_and(mask, std::memory_order_relaxed);
    }

    std::size_t bit_count_ = 0;
    std::size_t word_count_ = 0;
    std::unique_ptr<AtomicWord[]> words_;
};

} // namespace highvoronoi::detail
