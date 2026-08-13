#pragma once

#include "highvoronoi/detail/hash_generators.hpp"
#include "highvoronoi/detail/locks.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace highvoronoi::detail {

/**
 * @brief Probabilistic queue-membership table parameterized by its HashGenerator.
 *
 * HashGenerator must be default constructible and provide:
 *
 *   HashGenerator(key)
 *   bool operator==(const HashGenerator&, const HashGenerator&)
 *   size_t index(uint64_t mask, uint64_t i) const
 *
 * Lock is normally EmptyLock or ReadWriteLock.
 *
 * The Lock template parameter intentionally stays first so existing code such
 * as QueueHashTable<ReadWriteLock> remains source-compatible.
 */
template <
    class Lock = EmptyLock,
    class HashGenerator = FNV64_128HashGenerator
>
class QueueHashTable final {
public:
    using hash_code_type = HashGenerator;

    explicit QueueHashTable(std::size_t length = 1)
        : table_(next_power_of_two(length)),
          occupied_(table_.size(), 0),
          deleted_(table_.size(), 0),
          mask_(static_cast<std::uint64_t>(table_.size() - 1))
    {}

    QueueHashTable(const QueueHashTable&) = delete;
    QueueHashTable& operator=(const QueueHashTable&) = delete;
    QueueHashTable(QueueHashTable&&) = delete;
    QueueHashTable& operator=(QueueHashTable&&) = delete;

    template <class Key>
    [[nodiscard]] bool pushqueue(const Key& key, bool write = true)
    {
        // Hashing is intentionally outside the critical section, matching the
        // Julia implementation and keeping lock hold times short.
        const HashGenerator hash(key);

        if (write) {
            WriteLockGuard<Lock> guard(lock_);
            return insert_unlocked(hash);
        }

        ReadLockGuard<Lock> guard(lock_);
        return contains_unlocked(hash);
    }

    template <class Key>
    [[nodiscard]] bool contains(const Key& key) const
    {
        const HashGenerator hash(key);
        ReadLockGuard<Lock> guard(lock_);
        return contains_unlocked(hash);
    }

    /** Erase key if present. Returns true iff an entry was erased. */
    template <class Key>
    bool erase(const Key& key)
    {
        const HashGenerator hash(key);
        WriteLockGuard<Lock> guard(lock_);
        return erase_unlocked(hash);
    }

    /** Grow to at least length slots. Shrinking is intentionally ignored. */
    void resize(std::size_t length)
    {
        WriteLockGuard<Lock> guard(lock_);
        const std::size_t new_capacity = next_power_of_two(length);
        if (new_capacity > table_.size()) {
            rehash_unlocked(new_capacity);
        }
    }

    void clear()
    {
        WriteLockGuard<Lock> guard(lock_);
        std::fill(occupied_.begin(), occupied_.end(), std::uint8_t{0});
        std::fill(deleted_.begin(), deleted_.end(), std::uint8_t{0});
    }

    [[nodiscard]] std::size_t capacity() const
    {
        ReadLockGuard<Lock> guard(lock_);
        return table_.size();
    }

private:
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

    [[nodiscard]] bool contains_unlocked(const HashGenerator& hash) const
    {
        const std::uint64_t capacity64 =
            static_cast<std::uint64_t>(table_.size());

        for (std::uint64_t i = 0; i < capacity64; ++i) {
            const std::size_t idx = hash.index(mask_, i);

            if (occupied_[idx] != 0) {
                if (table_[idx] == hash) {
                    return true;
                }
                continue;
            }

            // A tombstone does not terminate a probe chain.
            if (deleted_[idx] != 0) {
                continue;
            }

            return false;
        }
        return false;
    }

    [[nodiscard]] bool insert_unlocked(const HashGenerator& hash)
    {
        for (;;) {
            std::size_t first_deleted = npos;
            const std::uint64_t capacity64 =
                static_cast<std::uint64_t>(table_.size());

            for (std::uint64_t i = 0; i < capacity64; ++i) {
                const std::size_t idx = hash.index(mask_, i);

                if (occupied_[idx] != 0) {
                    if (table_[idx] == hash) {
                        return true;
                    }
                    continue;
                }

                if (deleted_[idx] != 0) {
                    if (first_deleted == npos) {
                        first_deleted = idx;
                    }
                    continue;
                }

                // This slot has never been used. No matching code can occur
                // farther along this probe chain.
                const std::size_t target =
                    first_deleted == npos ? idx : first_deleted;
                table_[target] = hash;
                occupied_[target] = 1;
                deleted_[target] = 0;
                return false;
            }

            // The complete probe sequence was visited. Reuse a tombstone if
            // possible; otherwise grow and try again with the same HashGenerator.
            if (first_deleted != npos) {
                table_[first_deleted] = hash;
                occupied_[first_deleted] = 1;
                deleted_[first_deleted] = 0;
                return false;
            }

            rehash_unlocked(table_.size() * 2);
        }
    }

    bool erase_unlocked(const HashGenerator& hash)
    {
        const std::uint64_t capacity64 =
            static_cast<std::uint64_t>(table_.size());

        for (std::uint64_t i = 0; i < capacity64; ++i) {
            const std::size_t idx = hash.index(mask_, i);

            if (occupied_[idx] != 0) {
                if (table_[idx] == hash) {
                    occupied_[idx] = 0;
                    deleted_[idx] = 1;
                    return true;
                }
                continue;
            }

            if (deleted_[idx] != 0) {
                continue;
            }

            return false;
        }
        return false;
    }

    static void insert_rehashed(
        std::vector<HashGenerator>& table,
        std::vector<std::uint8_t>& occupied,
        std::uint64_t mask,
        const HashGenerator& hash)
    {
        const std::uint64_t capacity = mask + 1;

        for (std::uint64_t i = 0; i < capacity; ++i) {
            const std::size_t idx = hash.index(mask, i);
            if (occupied[idx] == 0) {
                table[idx] = hash;
                occupied[idx] = 1;
                return;
            }
        }

        throw std::logic_error(
            "QueueHashTable rehash failed to find a free slot");
    }

    void rehash_unlocked(std::size_t requested_capacity)
    {
        const std::size_t new_capacity = next_power_of_two(requested_capacity);
        std::vector<HashGenerator> new_table(new_capacity);
        std::vector<std::uint8_t> new_occupied(new_capacity, 0);
        const std::uint64_t new_mask =
            static_cast<std::uint64_t>(new_capacity - 1);

        for (std::size_t i = 0; i < table_.size(); ++i) {
            if (occupied_[i] != 0) {
                insert_rehashed(
                    new_table,
                    new_occupied,
                    new_mask,
                    table_[i]);
            }
        }

        table_.swap(new_table);
        occupied_.swap(new_occupied);
        deleted_.assign(new_capacity, 0);
        mask_ = new_mask;
    }

    std::vector<HashGenerator> table_;
    std::vector<std::uint8_t> occupied_;
    std::vector<std::uint8_t> deleted_;
    std::uint64_t mask_{0};
    mutable Lock lock_{};
};

/** No-op variant corresponding to Julia's EmptyQueueHashTable. */
class EmptyQueueHashTable final {
public:
    template <class Key>
    [[nodiscard]] constexpr bool pushqueue(const Key&, bool = true) const noexcept
    {
        return false;
    }

    template <class Key>
    [[nodiscard]] constexpr bool contains(const Key&) const noexcept
    {
        return false;
    }

    template <class Key>
    constexpr bool erase(const Key&) const noexcept
    {
        return false;
    }

    constexpr void resize(std::size_t) const noexcept {}
    constexpr void clear() const noexcept {}
};

} // namespace highvoronoi::detail
