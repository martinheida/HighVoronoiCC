#pragma once

#include "highvoronoi/storage/hash/hash_generators.hpp"
#include "highvoronoi/core/detail/locks.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace highvoronoi::detail {

/**
 * QueueHashTable_2
 *
 * Readers may continue while one writer inserts or erases.
 *
 * Lock order for writers:
 *   1. writelock_
 *   2. lock_.readlock()
 *
 * Rehashing:
 *   - writelock_ blocks all writers while the new table is built.
 *   - readers continue using the old table.
 *   - lock_.writelock() is taken only for the final swap.
 *
 * Published table entries are immutable until rehash/clear. Therefore deleted
 * slots are not overwritten immediately; rehash removes the tombstones.
 */
template <
    class Lock = EmptyLock,
    class HashGenerator = FNV64_128HashGenerator
>
class QueueHashTable_2 final {
public:
    using hash_generator_type = HashGenerator;

    explicit QueueHashTable_2(std::size_t length = 1)
        : table_(next_power_of_two(length)),
          occupied_(table_.size()),
          deleted_(table_.size()),
          mask_(static_cast<std::uint64_t>(table_.size() - 1))
    {
        reset_flags(occupied_);
        reset_flags(deleted_);
    }

    QueueHashTable_2(const QueueHashTable_2&) = delete;
    QueueHashTable_2& operator=(const QueueHashTable_2&) = delete;
    QueueHashTable_2(QueueHashTable_2&&) = delete;
    QueueHashTable_2& operator=(QueueHashTable_2&&) = delete;

    template <class Key>
    [[nodiscard]] bool pushqueue(const Key& key, bool write = true)
    {
        const HashGenerator hash(key);

        if (!write) {
            ReadLockGuard<Lock> guard(lock_);
            return contains_unlocked(hash);
        }

        for (;;) {
            std::size_t observed_capacity = 0;
            InsertResult result = InsertResult::full;

            // Important lock order: writer lock first, then table read lock.
            writelock_.writelock();
            lock_.readlock();

            try {
                observed_capacity = table_.size();
                result = insert_once_unlocked(hash);

                if (result == InsertResult::inserted) {
                    data_count_.fetch_add(1);
                }

                // Release the writer lock immediately after the mutation.
                writelock_.writeunlock();
                lock_.readunlock();
            }
            catch (...) {
                writelock_.writeunlock();
                lock_.readunlock();
                throw;
            }

            if (result == InsertResult::present) {
                return true;
            }

            if (result == InsertResult::inserted) {
                maybe_extend(observed_capacity);
                return false;
            }

            // No never-used slot was reachable. This can happen after many
            // deletions because deleted slots are intentionally not overwritten
            // while concurrent readers are allowed.
            force_rehash();
        }
    }

    template <class Key>
    [[nodiscard]] bool contains(const Key& key) const
    {
        const HashGenerator hash(key);
        ReadLockGuard<Lock> guard(lock_);
        return contains_unlocked(hash);
    }

    template <class Key>
    bool erase(const Key& key)
    {
        const HashGenerator hash(key);
        bool erased = false;

        writelock_.writelock();
        lock_.readlock();

        try {
            erased = erase_unlocked(hash);
            if (erased) {
                data_count_.fetch_sub(1);
            }

            writelock_.writeunlock();
            lock_.readunlock();
        }
        catch (...) {
            writelock_.writeunlock();
            lock_.readunlock();
            throw;
        }

        return erased;
    }

    void resize(std::size_t length)
    {
        writelock_.writelock();

        try {
            const std::size_t new_capacity = next_power_of_two(length);
            if (new_capacity > table_.size()) {
                rehash_with_writer_lock_held(new_capacity);
            }
            writelock_.writeunlock();
        }
        catch (...) {
            writelock_.writeunlock();
            throw;
        }
    }

    void clear()
    {
        // Same global order as rehash: writer lock first, table write lock second.
        writelock_.writelock();
        lock_.writelock();

        reset_flags(occupied_);
        reset_flags(deleted_);
        data_count_.store(0);

        lock_.writeunlock();
        writelock_.writeunlock();
    }

    [[nodiscard]] std::size_t capacity() const
    {
        ReadLockGuard<Lock> guard(lock_);
        return table_.size();
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return data_count_.load();
    }

private:
    enum class InsertResult {
        inserted,
        present,
        full
    };

    using AtomicFlag = std::atomic<std::uint8_t>;

    static void reset_flags(std::vector<AtomicFlag>& flags) noexcept
    {
        for (auto& flag : flags) {
            flag.store(0, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] bool contains_unlocked(const HashGenerator& hash) const
    {
        const std::uint64_t capacity64 =
            static_cast<std::uint64_t>(table_.size());

        for (std::uint64_t i = 0; i < capacity64; ++i) {
            const std::size_t idx = hash.index(mask_, i);

            if (occupied_[idx].load(std::memory_order_acquire) == 0) {
                // This slot has never been published, so the probe chain ends.
                return false;
            }

            if (deleted_[idx].load(std::memory_order_acquire) != 0) {
                continue;
            }

            if (table_[idx] == hash) {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] InsertResult insert_once_unlocked(
        const HashGenerator& hash)
    {
        const std::uint64_t capacity64 =
            static_cast<std::uint64_t>(table_.size());

        for (std::uint64_t i = 0; i < capacity64; ++i) {
            const std::size_t idx = hash.index(mask_, i);

            if (occupied_[idx].load(std::memory_order_acquire) == 0) {
                // Write the immutable payload first, publish the slot last.
                table_[idx] = hash;
                deleted_[idx].store(0, std::memory_order_relaxed);
                occupied_[idx].store(1, std::memory_order_release);
                return InsertResult::inserted;
            }

            if (deleted_[idx].load(std::memory_order_acquire) != 0) {
                continue;
            }

            if (table_[idx] == hash) {
                return InsertResult::present;
            }
        }

        return InsertResult::full;
    }

    [[nodiscard]] bool erase_unlocked(const HashGenerator& hash)
    {
        const std::uint64_t capacity64 =
            static_cast<std::uint64_t>(table_.size());

        for (std::uint64_t i = 0; i < capacity64; ++i) {
            const std::size_t idx = hash.index(mask_, i);

            if (occupied_[idx].load(std::memory_order_acquire) == 0) {
                return false;
            }

            if (deleted_[idx].load(std::memory_order_acquire) != 0) {
                continue;
            }

            if (table_[idx] == hash) {
                // The hash payload stays untouched. Readers that already saw
                // the old state may still finish safely.
                deleted_[idx].store(1, std::memory_order_release);
                return true;
            }
        }

        return false;
    }

    static void insert_rehashed(
        std::vector<HashGenerator>& table,
        std::vector<AtomicFlag>& occupied,
        std::uint64_t mask,
        const HashGenerator& hash)
    {
        const std::uint64_t capacity = mask + 1;

        for (std::uint64_t i = 0; i < capacity; ++i) {
            const std::size_t idx = hash.index(mask, i);

            if (occupied[idx].load(std::memory_order_relaxed) == 0) {
                table[idx] = hash;
                occupied[idx].store(1, std::memory_order_relaxed);
                return;
            }
        }

        throw std::logic_error(
            "QueueHashTable_2 rehash failed to find a free slot");
    }

    /**
     * Caller must hold writelock_.
     *
     * No writer can modify the old table while the new table is built.
     * Readers may continue using it until the short final swap.
     */
    void rehash_with_writer_lock_held(std::size_t requested_capacity)
    {
        const std::size_t new_capacity =
            next_power_of_two(requested_capacity);

        std::vector<HashGenerator> new_table(new_capacity);
        std::vector<AtomicFlag> new_occupied(new_capacity);
        std::vector<AtomicFlag> new_deleted(new_capacity);

        reset_flags(new_occupied);
        reset_flags(new_deleted);

        const std::uint64_t new_mask =
            static_cast<std::uint64_t>(new_capacity - 1);

        for (std::size_t i = 0; i < table_.size(); ++i) {
            if (occupied_[i].load(std::memory_order_acquire) != 0 &&
                deleted_[i].load(std::memory_order_acquire) == 0) {
                insert_rehashed(
                    new_table,
                    new_occupied,
                    new_mask,
                    table_[i]);
            }
        }

        // Readers are stopped only for the actual table replacement.
        lock_.writelock();

        table_.swap(new_table);
        occupied_.swap(new_occupied);
        deleted_.swap(new_deleted);
        mask_ = new_mask;

        lock_.writeunlock();
    }

    /**
     * Called after a successful insertion.
     *
     * Only one thread performs the extension. Other writers do not wait for
     * the rehash merely because the half-full threshold was crossed.
     */
    void maybe_extend(std::size_t observed_capacity)
    {
        if (data_count_.load() <= observed_capacity / 2) {
            return;
        }

        const int thread_count = extending_.fetch_add(1);

        if (thread_count > 0) {
            extending_.fetch_sub(1);
            return;
        }

        try {
            writelock_.writelock();

            try {
                // Revalidate because another extension may already have made
                // the old half-full observation obsolete.
                if (data_count_.load() > table_.size() / 2) {
                    rehash_with_writer_lock_held(table_.size() * 2);
                }

                writelock_.writeunlock();
            }
            catch (...) {
                writelock_.writeunlock();
                throw;
            }

            extending_.fetch_sub(1);
        }
        catch (...) {
            extending_.fetch_sub(1);
            throw;
        }
    }

    /**
     * Required only when no unpublished slot can be found, usually because
     * tombstones accumulated. Unlike maybe_extend(), the failed insertion must
     * wait until the active rehash is finished and then retry.
     */
    void force_rehash()
    {
        for (;;) {
            const int thread_count = extending_.fetch_add(1);

            if (thread_count > 0) {
                extending_.fetch_sub(1);

                while (extending_.load() != 0) {
                    std::this_thread::yield();
                }
                return;
            }

            try {
                writelock_.writelock();

                try {
                    const std::size_t current_capacity = table_.size();
                    const std::size_t target_capacity =
                        data_count_.load() > current_capacity / 2
                            ? current_capacity * 2
                            : current_capacity;

                    rehash_with_writer_lock_held(target_capacity);
                    writelock_.writeunlock();
                }
                catch (...) {
                    writelock_.writeunlock();
                    throw;
                }

                extending_.fetch_sub(1);
                return;
            }
            catch (...) {
                extending_.fetch_sub(1);
                throw;
            }
        }
    }

    std::vector<HashGenerator> table_;
    std::vector<AtomicFlag> occupied_;
    std::vector<AtomicFlag> deleted_;

    std::uint64_t mask_{0};

    mutable Lock lock_{};
    mutable Lock writelock_{};

    std::atomic<int> extending_{0};
    std::atomic<std::size_t> data_count_{0};
};

} // namespace highvoronoi::detail
