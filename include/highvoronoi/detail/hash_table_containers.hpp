#pragma once

#include "highvoronoi/detail/locks.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace highvoronoi::detail {

namespace hash_container_detail {

template<class Key>
[[nodiscard]] std::uint64_t first_value(const Key& key)
{
    return key[0];
}

template<class HashTable, std::size_t N, std::size_t... I>
[[nodiscard]] std::array<HashTable, N> make_hash_array(
    std::size_t hash_capacity,
    std::index_sequence<I...>)
{
    return {
        ((void)I, HashTable(hash_capacity))...
    };
}

template<class HashTable, std::size_t N>
[[nodiscard]] std::array<HashTable, N> make_hash_array(
    std::size_t hash_capacity)
{
    return make_hash_array<HashTable, N>(
        hash_capacity,
        std::make_index_sequence<N>{});
}

} // namespace hash_container_detail


// ============================================================================
// Dynamic queue container
//
// Routing:
//   table index = key[0] / block_size
//
// The vector grows when the required table does not yet exist.
// ResizeLock protects only the vector structure.
// Each QueueHashTable protects its own contents.
// ============================================================================

template<class QueueHashTable, class ResizeLock = ReadWriteLock>
class DynamicQueueHashContainer final {
public:
    explicit DynamicQueueHashContainer(
        std::size_t initial_table_count,
        std::uint64_t block_size,
        std::size_t hash_capacity = 1)
        : block_size_(block_size),
          hash_capacity_(hash_capacity)
    {
        if (block_size_ == 0) {
            throw std::invalid_argument(
                "DynamicQueueHashContainer block_size must be greater than zero");
        }

        tables_.reserve(initial_table_count);

        for (std::size_t i = 0; i < initial_table_count; ++i) {
            tables_.push_back(
                std::make_unique<QueueHashTable>(hash_capacity_));
        }

        length_ = tables_.size();
    }

    template<class Key>
    [[nodiscard]] bool pushqueue(const Key& key, bool write = true)
    {
        const std::size_t index = table_index(key);

        if (!write) {
            QueueHashTable* table = find_table(index);
            return table != nullptr && table->pushqueue(key, false);
        }

        return ensure_table(index)->pushqueue(key, true);
    }

    template<class Key>
    [[nodiscard]] bool contains(const Key& key) const
    {
        const std::size_t index = table_index(key);
        const QueueHashTable* table = find_table(index);

        return table != nullptr && table->contains(key);
    }

    template<class Key>
    bool erase(const Key& key)
    {
        const std::size_t index = table_index(key);
        QueueHashTable* table = find_table(index);

        return table != nullptr && table->erase(key);
    }

    void clear()
    {
        ReadLockGuard<ResizeLock> guard(resize_lock_);

        for (const auto& table : tables_) {
            table->clear();
        }
    }

    [[nodiscard]] std::size_t table_count() const
    {
        ReadLockGuard<ResizeLock> guard(resize_lock_);
        return length_;
    }

    [[nodiscard]] std::uint64_t block_size() const noexcept
    {
        return block_size_;
    }

private:
    template<class Key>
    [[nodiscard]] std::size_t table_index(const Key& key) const
    {
        return hash_container_detail::first_value(key) / block_size_;
    }

    [[nodiscard]] QueueHashTable* find_table(std::size_t index)
    {
        ReadLockGuard<ResizeLock> guard(resize_lock_);

        if (index >= length_) {
            return nullptr;
        }

        return tables_[index].get();
    }

    [[nodiscard]] const QueueHashTable* find_table(
        std::size_t index) const
    {
        ReadLockGuard<ResizeLock> guard(resize_lock_);

        if (index >= length_) {
            return nullptr;
        }

        return tables_[index].get();
    }

    [[nodiscard]] QueueHashTable* ensure_table(std::size_t index)
    {
        {
            ReadLockGuard<ResizeLock> guard(resize_lock_);

            if (index < length_) {
                return tables_[index].get();
            }
        }

        WriteLockGuard<ResizeLock> guard(resize_lock_);

        while (length_ <= index) {
            tables_.push_back(
                std::make_unique<QueueHashTable>(hash_capacity_));
            ++length_;
        }

        return tables_[index].get();
    }

    std::vector<std::unique_ptr<QueueHashTable>> tables_;
    std::size_t length_{0};
    std::uint64_t block_size_{1};
    std::size_t hash_capacity_{1};
    mutable ResizeLock resize_lock_{};
};


// ============================================================================
// Fixed queue container
//
// Routing:
//   table index = key[0] % N
//
// No outer lock is needed because the array never changes its size.
// Each QueueHashTable protects its own contents.
// ============================================================================

template<class QueueHashTable, std::size_t N>
class StaticQueueHashContainer final {
    static_assert(N > 0, "StaticQueueHashContainer requires N > 0");

public:
    explicit StaticQueueHashContainer(std::size_t hash_capacity = 1)
        : tables_(
            hash_container_detail::make_hash_array<QueueHashTable, N>(
                hash_capacity))
    {}

    template<class Key>
    [[nodiscard]] bool pushqueue(const Key& key, bool write = true)
    {
        return tables_[table_index(key)].pushqueue(key, write);
    }

    template<class Key>
    [[nodiscard]] bool contains(const Key& key) const
    {
        return tables_[table_index(key)].contains(key);
    }

    template<class Key>
    bool erase(const Key& key)
    {
        return tables_[table_index(key)].erase(key);
    }

    void clear()
    {
        for (auto& table : tables_) {
            table.clear();
        }
    }

    [[nodiscard]] static constexpr std::size_t table_count() noexcept
    {
        return N;
    }

private:
    template<class Key>
    [[nodiscard]] static std::size_t table_index(const Key& key)
    {
        return hash_container_detail::first_value(key) % N;
    }

    std::array<QueueHashTable, N> tables_;
};


// ============================================================================
// Dynamic edge container
//
// Routing:
//   table index = key[0] / block_size
//
// The vector grows when the required table does not yet exist.
// ResizeLock protects only the vector structure.
// Each EdgeHashTable protects its own contents.
// ============================================================================

template<class EdgeHashTable, class ResizeLock = ReadWriteLock>
class DynamicEdgeHashContainer final {
public:
    static constexpr std::int64_t infinite_cell =
        EdgeHashTable::infinite_cell;

    explicit DynamicEdgeHashContainer(
        std::size_t initial_table_count,
        std::uint64_t block_size,
        std::size_t hash_capacity = 1)
        : block_size_(block_size),
          hash_capacity_(hash_capacity)
    {
        if (block_size_ == 0) {
            throw std::invalid_argument(
                "DynamicEdgeHashContainer block_size must be greater than zero");
        }

        tables_.reserve(initial_table_count);

        for (std::size_t i = 0; i < initial_table_count; ++i) {
            tables_.push_back(
                std::make_unique<EdgeHashTable>(hash_capacity_));
        }

        length_ = tables_.size();
    }

    template<class Key>
    [[nodiscard]] bool pushedge(
        const Key& key,
        std::int64_t cell,
        bool mode = true)
    {
        const std::size_t index = table_index(key);
        return ensure_table(index)->pushedge(key, cell, mode);
    }

    void clear()
    {
        ReadLockGuard<ResizeLock> guard(resize_lock_);

        for (const auto& table : tables_) {
            table->clear();
        }
    }

    [[nodiscard]] bool all_edges_complete() const
    {
        ReadLockGuard<ResizeLock> guard(resize_lock_);
        for (const auto& table : tables_) {
            if (!table->all_edges_complete()) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::size_t table_count() const
    {
        ReadLockGuard<ResizeLock> guard(resize_lock_);
        return length_;
    }

    [[nodiscard]] std::uint64_t block_size() const noexcept
    {
        return block_size_;
    }

private:
    template<class Key>
    [[nodiscard]] std::size_t table_index(const Key& key) const
    {
        return hash_container_detail::first_value(key) / block_size_;
    }

    [[nodiscard]] EdgeHashTable* ensure_table(std::size_t index)
    {
        {
            ReadLockGuard<ResizeLock> guard(resize_lock_);

            if (index < length_) {
                return tables_[index].get();
            }
        }

        WriteLockGuard<ResizeLock> guard(resize_lock_);

        while (length_ <= index) {
            tables_.push_back(
                std::make_unique<EdgeHashTable>(hash_capacity_));
            ++length_;
        }

        return tables_[index].get();
    }

    std::vector<std::unique_ptr<EdgeHashTable>> tables_;
    std::size_t length_{0};
    std::uint64_t block_size_{1};
    std::size_t hash_capacity_{1};
    mutable ResizeLock resize_lock_{};
};


// ============================================================================
// Fixed edge container
//
// Routing:
//   table index = key[0] % N
//
// No outer lock is needed because the array never changes its size.
// Each EdgeHashTable protects its own contents.
// ============================================================================

template<class EdgeHashTable, std::size_t N>
class StaticEdgeHashContainer final {
    static_assert(N > 0, "StaticEdgeHashContainer requires N > 0");

public:
    static constexpr std::int64_t infinite_cell =
        EdgeHashTable::infinite_cell;

    explicit StaticEdgeHashContainer(std::size_t hash_capacity = 1)
        : tables_(
            hash_container_detail::make_hash_array<EdgeHashTable, N>(
                hash_capacity))
    {}

    template<class Key>
    [[nodiscard]] bool pushedge(
        const Key& key,
        std::int64_t cell,
        bool mode = true)
    {
        return tables_[table_index(key)].pushedge(key, cell, mode);
    }

    void clear()
    {
        for (auto& table : tables_) {
            table.clear();
        }
    }

    [[nodiscard]] bool all_edges_complete() const
    {
        for (const auto& table : tables_) {
            if (!table.all_edges_complete()) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static constexpr std::size_t table_count() noexcept
    {
        return N;
    }

private:
    template<class Key>
    [[nodiscard]] static std::size_t table_index(const Key& key)
    {
        return hash_container_detail::first_value(key) % N;
    }

    std::array<EdgeHashTable, N> tables_;
};

} // namespace highvoronoi::detail
