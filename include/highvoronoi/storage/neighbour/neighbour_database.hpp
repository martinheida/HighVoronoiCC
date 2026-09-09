#pragma once

/**
 * @file neighbour_database.hpp
 * @brief Append-only block database for variable-length Voronoi neighbour lists.
 *
 * The storage and synchronization model intentionally follows HVDataBase:
 *
 * - storage is a vector of fixed-size blocks of 16-bit units;
 * - an atomic top counter reserves disjoint record ranges for concurrent writers;
 * - the outer block vector is exclusively locked only when it must grow;
 * - once sufficient capacity exists, readers and writers hold only a shared
 *   read lock so the block structure cannot move while disjoint payload ranges
 *   are accessed;
 * - records are append-only and never reclaimed.
 *
 * There is deliberately no hash table and no duplicate suppression. One record
 * has the layout
 *
 *     std::size_t length, Index neighbour[0], ..., Index neighbour[length-1]
 *
 * The length therefore does not inherit the width of Index and is not limited
 * to uint16_t. Input ordering and duplicate neighbour entries are preserved.
 */

#include <highvoronoi/core/detail/atomic_bit_vector.hpp>
#include <highvoronoi/core/detail/locks.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi::detail {

/**
 * @brief Dirty-bit representation selected from the mesh/database lock policy.
 *
 * EmptyLock denotes a structurally single-threaded mesh and uses the cheaper
 * packed BitVector. Every real synchronization lock selects AtomicBitVector so
 * different workers may set/reset/test dirty bits concurrently.
 */
template <class Lock>
using NeighbourDirtyVector = std::conditional_t<
    std::is_same_v<Lock, EmptyLock>,
    BitVector,
    AtomicBitVector>;

/**
 * @brief Lock-aware append-only storage for neighbour-index vectors.
 *
 * @tparam Lock   Same lock policy used by the owning mesh/database.
 * @tparam IndexT Mesh index type stored in neighbour records.
 */
template <class Lock, class IndexT>
class NeighbourDatabase final {
public:
    using size_type = std::size_t;
    using UInt16 = std::uint16_t;
    using Index = IndexT;
    using address_type = std::size_t;
    using DataUnit = std::vector<UInt16>;

    using LockType = Lock;
    using ReadGuard = ReadLockGuard<Lock>;
    using WriteGuard = WriteLockGuard<Lock>;

    static constexpr size_type index_unit_count =
        sizeof(Index) / sizeof(UInt16);
    static constexpr size_type length_unit_count =
        sizeof(size_type) / sizeof(UInt16);

    static_assert(
        std::is_integral_v<Index> &&
        !std::is_same_v<std::remove_cv_t<Index>, bool>,
        "NeighbourDatabase Index must be a non-bool integral type");
    static_assert(
        sizeof(Index) == 2 || sizeof(Index) == 4 || sizeof(Index) == 8,
        "NeighbourDatabase Index must have 16, 32, or 64 bits");
    static_assert(
        std::is_trivially_copyable_v<Index>,
        "NeighbourDatabase Index must be trivially copyable");
    static_assert(
        sizeof(size_type) % sizeof(UInt16) == 0,
        "NeighbourDatabase size_t must consist of complete 16-bit units");

    explicit NeighbourDatabase(size_type unit_length = size_type{65536})
        : unit_length_(checked_unit_length(unit_length)) {}

    NeighbourDatabase(const NeighbourDatabase&) = delete;
    NeighbourDatabase& operator=(const NeighbourDatabase&) = delete;
    NeighbourDatabase(NeighbourDatabase&&) = delete;
    NeighbourDatabase& operator=(NeighbourDatabase&&) = delete;

    /**
     * @brief Append one complete neighbour list and return its one-based address.
     *
     * Empty vectors are valid records. No sorting, unique operation or hash
     * lookup is performed here.
     */
    template <class IndexVector>
    [[nodiscard]] address_type push(const IndexVector& neighbours) {
        require_index_vector<IndexVector>();

        const size_type count = static_cast<size_type>(neighbours.size());
        if (count >
            ((std::numeric_limits<size_type>::max)() - length_unit_count) /
                index_unit_count) {
            throw std::overflow_error(
                "NeighbourDatabase record size exceeds addressable storage");
        }

        const size_type units =
            length_unit_count + index_unit_count * count;
        const size_type begin = reserve_units(units);

        if (begin > (std::numeric_limits<size_type>::max)() - units) {
            throw std::overflow_error(
                "NeighbourDatabase stream position overflow");
        }

        ensure_capacity(begin + units);

        // Same contention pattern as HVDataBase: capacity changes require the
        // exclusive lock; ordinary writes only protect the stability of data_.
        // Each writer owns a disjoint range reserved through top_.fetch_add().
        ReadGuard guard(lock_);
        size_type position = begin;
        write_value(position, count);
        write_values(position, neighbours.data(), count);

        return begin + address_type{1};
    }

    /**
     * @brief Read one record into caller-owned recyclable storage.
     *
     * Address zero means "no record" and clears the destination.
     */
    template <class IndexVector>
    [[nodiscard]] bool read(
        address_type address,
        IndexVector& neighbours) const {
        require_mutable_index_vector<IndexVector>();
        neighbours.clear();

        if (address == address_type{0}) {
            return false;
        }

        ReadGuard guard(lock_);
        size_type position = checked_position(address);
        const size_type count = read_value<size_type>(position);
        require_record_extent(position, count);

        neighbours.resize(count);
        read_values(position, neighbours.data(), count);
        return true;
    }

    [[nodiscard]] size_type reserved_unit_count() const noexcept {
        return top_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_type block_count() const {
        ReadGuard guard(lock_);
        return data_.size();
    }

    [[nodiscard]] size_type block_unit_length() const noexcept {
        return unit_length_;
    }

private:
    template <class Vector>
    using DataElement = std::remove_cv_t<
        std::remove_pointer_t<decltype(std::declval<Vector&>().data())>>;

    template <class Vector>
    static constexpr void require_index_vector() {
        static_assert(
            std::is_same_v<DataElement<Vector>, Index>,
            "NeighbourDatabase vectors must contain Index and provide data()");
    }

    template <class Vector>
    static constexpr void require_mutable_index_vector() {
        require_index_vector<Vector>();
        static_assert(
            !std::is_const_v<
                std::remove_pointer_t<decltype(std::declval<Vector&>().data())>>,
            "NeighbourDatabase destination must provide mutable Index storage");
    }

    [[nodiscard]] static size_type checked_unit_length(size_type unit_length) {
        if (unit_length == size_type{0}) {
            throw std::invalid_argument(
                "NeighbourDatabase block length must be greater than zero");
        }
        return unit_length;
    }

    /** Reserve an exclusive stream interval exactly as HVDataBase does. */
    [[nodiscard]] size_type reserve_units(size_type count) noexcept {
        return top_.fetch_add(count, std::memory_order_relaxed);
    }

    /** Grow only the outer block structure under the exclusive lock. */
    void ensure_capacity(size_type required_units) {
        if (capacity_.load(std::memory_order_acquire) >= required_units) {
            return;
        }

        WriteGuard guard(lock_);

        if (capacity_.load(std::memory_order_relaxed) >= required_units) {
            return;
        }

        const size_type required_blocks =
            (required_units + unit_length_ - size_type{1}) / unit_length_;
        const size_type old_block_count = data_.size();

        data_.resize(required_blocks);
        for (size_type i = old_block_count; i < required_blocks; ++i) {
            data_[i].resize(unit_length_);
        }

        capacity_.store(
            required_blocks * unit_length_,
            std::memory_order_release);
    }

    [[nodiscard]] size_type checked_position(address_type address) const {
        const size_type position = address - address_type{1};
        if (position >= top_.load(std::memory_order_acquire)) {
            throw std::out_of_range(
                "NeighbourDatabase address is outside reserved storage");
        }
        return position;
    }

    void require_record_extent(
        size_type payload_position,
        size_type count) const {
        if (count >
            (std::numeric_limits<size_type>::max)() / index_unit_count) {
            throw std::runtime_error(
                "NeighbourDatabase contains an invalid record length");
        }

        const size_type payload_units = index_unit_count * count;
        const size_type reserved = top_.load(std::memory_order_acquire);
        if (payload_position > reserved ||
            payload_units > reserved - payload_position) {
            throw std::runtime_error(
                "NeighbourDatabase contains a truncated record");
        }
    }

    /** Copy a contiguous value range into the block stream. */
    template <class Value>
    void write_values(
        size_type& position,
        const Value* source,
        size_type count) {
        static_assert(
            sizeof(Value) % sizeof(UInt16) == 0,
            "Stored values must consist of complete 16-bit units");

        const auto* source_bytes =
            reinterpret_cast<const unsigned char*>(source);
        size_type remaining_units =
            count * sizeof(Value) / sizeof(UInt16);

        while (remaining_units > size_type{0}) {
            const size_type block = position / unit_length_;
            const size_type offset = position % unit_length_;
            const size_type copied_units = std::min(
                remaining_units,
                unit_length_ - offset);
            const size_type copied_bytes =
                copied_units * sizeof(UInt16);

            std::memcpy(
                data_[block].data() + offset,
                source_bytes,
                copied_bytes);

            source_bytes += copied_bytes;
            position += copied_units;
            remaining_units -= copied_units;
        }
    }

    /** Copy a contiguous value range from the block stream. */
    template <class Value>
    void read_values(
        size_type& position,
        Value* destination,
        size_type count) const {
        static_assert(
            sizeof(Value) % sizeof(UInt16) == 0,
            "Stored values must consist of complete 16-bit units");

        auto* destination_bytes =
            reinterpret_cast<unsigned char*>(destination);
        size_type remaining_units =
            count * sizeof(Value) / sizeof(UInt16);

        while (remaining_units > size_type{0}) {
            const size_type block = position / unit_length_;
            const size_type offset = position % unit_length_;
            const size_type copied_units = std::min(
                remaining_units,
                unit_length_ - offset);
            const size_type copied_bytes =
                copied_units * sizeof(UInt16);

            std::memcpy(
                destination_bytes,
                data_[block].data() + offset,
                copied_bytes);

            destination_bytes += copied_bytes;
            position += copied_units;
            remaining_units -= copied_units;
        }
    }

    template <class Value>
    void write_value(size_type& position, const Value& value) {
        write_values(position, &value, size_type{1});
    }

    template <class Value>
    [[nodiscard]] Value read_value(size_type& position) const {
        Value value{};
        read_values(position, &value, size_type{1});
        return value;
    }

    mutable Lock lock_{};
    std::vector<DataUnit> data_{};
    const size_type unit_length_;
    std::atomic<size_type> top_{0};
    std::atomic<size_type> capacity_{0};
};

} // namespace highvoronoi::detail
