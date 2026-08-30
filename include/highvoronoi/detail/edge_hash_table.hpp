
#pragma once

#include "highvoronoi/detail/hash_generators.hpp"
#include "highvoronoi/detail/locks.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace highvoronoi::detail {

template <class HashGenerator>
struct HashedEdge final {
    static constexpr std::int64_t no_second_cell = -1;
    static constexpr std::int64_t infinite_cell = -2;

    HashGenerator hash{};
    std::int64_t cell1{0};
    std::int64_t cell2{no_second_cell};
};

/**
 * @brief Probabilistic edge table parameterized by its HashGenerator.
 *
 * The table itself knows nothing about hash widths or hash functions. It only
 * stores HashGenerator objects, compares them, and asks them for probe positions.
 * Every pushedge operation mutates the table and therefore takes the write lock.
 *
 * Lock stays the first template parameter for source compatibility with the
 * previous EdgeHashTable<ReadWriteLock> spelling.
 */
template <
    class Lock = EmptyLock,
    class HashGenerator = FNV64_128HashGenerator
>
class EdgeHashTable final {
public:
    using hash_code_type = HashGenerator;
    using entry_type = HashedEdge<HashGenerator>;
    static constexpr std::int64_t infinite_cell =
        entry_type::infinite_cell;

    explicit EdgeHashTable(std::size_t length = 1)
        : table_(next_power_of_two(length)),
          occupied_(table_.size(), 0),
          mask_(static_cast<std::uint64_t>(table_.size() - 1))
    {}

    EdgeHashTable(const EdgeHashTable&) = delete;
    EdgeHashTable& operator=(const EdgeHashTable&) = delete;
    EdgeHashTable(EdgeHashTable&&) = delete;
    EdgeHashTable& operator=(EdgeHashTable&&) = delete;

    /**
     * @return true iff the matching edge already had a second cell.
     */
    template <class Key>
    [[nodiscard]] bool pushedge(
        const Key& key,
        std::int64_t cell,
        bool mode = true)
    {
        // Hashing does not mutate the table and is deliberately performed
        // before acquiring the exclusive lock.
        const HashGenerator hash(key);
        WriteLockGuard<Lock> guard(lock_);
        return pushedge_unlocked(hash, cell, mode);
    }

    void clear()
    {
        WriteLockGuard<Lock> guard(lock_);
        std::fill(occupied_.begin(), occupied_.end(), std::uint8_t{0});
        overfull_ = false;
    }

    /**
     * @brief Test whether an edge key is already present without mutating the table.
     *
     * The lookup uses the same probabilistic HashGenerator identity as pushedge().
     * No deletion tombstones exist in EdgeHashTable, therefore probing may stop at
     * the first unoccupied slot.
     */
    template <class Key>
    [[nodiscard]] bool contains(const Key& key) const
    {
        const HashGenerator hash(key);
        ReadLockGuard<Lock> guard(lock_);

        const std::uint64_t capacity64 =
            static_cast<std::uint64_t>(table_.size());

        for (std::uint64_t i = 0; i < capacity64; ++i) {
            const std::size_t idx = hash.index(mask_, i);

            if (occupied_[idx] == 0) {
                return false;
            }

            if (table_[idx].hash == hash) {
                return true;
            }
        }

        return false;
    }

    /**
     * @brief Return true iff every stored edge was encountered exactly twice.
     *
     * A third or later occurrence is remembered by pushedge() and makes this
     * check fail. This is intended for global mesh-completeness diagnostics;
     * the construction algorithm may continue to use the pushedge() return
     * value exactly as before.
     */
    [[nodiscard]] bool all_edges_complete() const
    {
        ReadLockGuard<Lock> guard(lock_);

        if (overfull_) {
            return false;
        }

        for (std::size_t i = 0; i < table_.size(); ++i) {
            if (occupied_[i] != 0 &&
                table_[i].cell2 == entry_type::no_second_cell) {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] std::size_t capacity() const
    {
        ReadLockGuard<Lock> guard(lock_);
        return table_.size();
    }

private:
    [[nodiscard]] bool pushedge_unlocked(
        const HashGenerator& hash,
        std::int64_t cell,
        bool mode)
    {
        for (;;) {
            const std::uint64_t capacity64 =
                static_cast<std::uint64_t>(table_.size());

            for (std::uint64_t i = 0; i < capacity64; ++i) {
                const std::size_t idx = hash.index(mask_, i);

                if (occupied_[idx] == 0) {
                    table_[idx] = entry_type{
                        hash,
                        cell,
                        entry_type::no_second_cell};
                    occupied_[idx] = 1;
                    return false;
                }

                entry_type& current = table_[idx];
                if (current.hash != hash) {
                    continue;
                }

                if (current.cell2 != entry_type::no_second_cell) {
                    overfull_ = true;
                    return true;
                }

                if (mode || current.cell1 != cell) {
                    current.cell2 = cell;
                }
                return false;
            }

            rehash_unlocked(table_.size() * 2);
        }
    }

    static void insert_rehashed(
        std::vector<entry_type>& table,
        std::vector<std::uint8_t>& occupied,
        std::uint64_t mask,
        const entry_type& value)
    {
        const std::uint64_t capacity = mask + 1;

        for (std::uint64_t i = 0; i < capacity; ++i) {
            const std::size_t idx = value.hash.index(mask, i);
            if (occupied[idx] == 0) {
                table[idx] = value;
                occupied[idx] = 1;
                return;
            }
        }

        throw std::logic_error(
            "EdgeHashTable rehash failed to find a free slot");
    }

    void rehash_unlocked(std::size_t requested_capacity)
    {
        const std::size_t new_capacity = next_power_of_two(requested_capacity);
        std::vector<entry_type> new_table(new_capacity);
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
        mask_ = new_mask;
    }

    std::vector<entry_type> table_;
    std::vector<std::uint8_t> occupied_;
    std::uint64_t mask_{0};
    bool overfull_{false};
    mutable Lock lock_{};
};

} // namespace highvoronoi::detail
