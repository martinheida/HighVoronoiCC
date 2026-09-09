#pragma once

/**
 * @file numeric_database.hpp
 * @brief Append-only block storage for variable-length floating-point records.
 *
 * This is the small storage primitive shared by AreaDatabase and
 * IntegralDatabase. It deliberately has no hash and no semantic knowledge.
 * Records preserve order and multiplicity exactly.
 */

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

template <class Lock, class FloatT>
class NumericDatabase {
public:
    using value_type = FloatT;
    using size_type = std::size_t;
    using address_type = std::size_t;
    using UInt16 = std::uint16_t;
    using DataUnit = std::vector<UInt16>;
    using LockType = Lock;
    using ReadGuard = ReadLockGuard<Lock>;
    using WriteGuard = WriteLockGuard<Lock>;

    static constexpr size_type value_unit_count =
        sizeof(value_type) / sizeof(UInt16);
    static constexpr size_type length_unit_count =
        sizeof(size_type) / sizeof(UInt16);

    static_assert(
        std::is_floating_point_v<value_type>,
        "NumericDatabase value_type must be a floating-point type");
    static_assert(
        std::is_trivially_copyable_v<value_type>,
        "NumericDatabase value_type must be trivially copyable");
    static_assert(
        sizeof(value_type) % sizeof(UInt16) == 0,
        "NumericDatabase value_type must consist of complete 16-bit units");
    static_assert(
        sizeof(size_type) % sizeof(UInt16) == 0,
        "NumericDatabase size_t must consist of complete 16-bit units");

    explicit NumericDatabase(size_type unit_length = size_type{65536})
        : unit_length_(checked_unit_length(unit_length)) {}

    NumericDatabase(const NumericDatabase&) = delete;
    NumericDatabase& operator=(const NumericDatabase&) = delete;
    NumericDatabase(NumericDatabase&&) = delete;
    NumericDatabase& operator=(NumericDatabase&&) = delete;

    /** Append one complete record and return its one-based physical address. */
    template <class ValueVector>
    address_type push(const ValueVector& values) {
        require_value_vector<ValueVector>();

        // Physical record layout: [record length][contiguous floating-point payload].
        // Both parts are serialized into 16-bit units so records may cross storage blocks.
        const size_type count = static_cast<size_type>(values.size());
        if (count >
            ((std::numeric_limits<size_type>::max)() - length_unit_count) /
                value_unit_count) {
            throw std::overflow_error(
                "NumericDatabase record size exceeds addressable storage");
        }

        const size_type units =
            length_unit_count + value_unit_count * count;
        // Reserve a unique append-only stream range first. Capacity growth is handled
        // separately, so concurrent writers never compete for one logical record slot.
        const size_type begin = reserve_units(units);

        if (begin > (std::numeric_limits<size_type>::max)() - units) {
            throw std::overflow_error(
                "NumericDatabase stream position overflow");
        }

        ensure_capacity(begin + units);

        ReadGuard guard(lock_);
        size_type position = begin;
        write_value(position, count);
        write_values(position, values.data(), count);
        return begin + address_type{1};
    }

    /** Convenience overload for one scalar record. */
    address_type push_scalar(value_type value) {
        const value_type values[1]{value};
        return push_contiguous(values, size_type{1});
    }

    /** Read one record into caller-owned recyclable storage. */
    template <class ValueVector>
    bool read(address_type address, ValueVector& values) const {
        require_mutable_value_vector<ValueVector>();
        values.clear();

        if (address == address_type{0}) {
            return false;
        }

        // Convert the public one-based physical address back to the stream position,
        // read the stored record length, then reconstruct the caller-owned vector.
        ReadGuard guard(lock_);
        size_type position = checked_position(address);
        const size_type count = read_value<size_type>(position);
        require_record_extent(position, count);

        values.resize(count);
        read_values(position, values.data(), count);
        return true;
    }


    /**
     * @brief Overwrite an existing record without changing its address or length.
     *
     * This is the only sanctioned in-place mutation for numeric records. It is
     * intended for the serial cleanup phase of an integration pass, where the
     * freshly created area/interface records may be symmetrized after the first
     * cell pass. The stored length word is read first and must match the new
     * value count exactly; otherwise no payload is written and an exception is
     * thrown.
     */
    template <class ValueVector>
    void overwrite(address_type address, const ValueVector& values) {
        require_value_vector<ValueVector>();
        if (address == address_type{0}) {
            throw std::invalid_argument(
                "NumericDatabase cannot overwrite address zero");
        }

        WriteGuard guard(lock_);
        size_type header_position = checked_position(address);
        size_type payload_position = header_position;
        const size_type old_count = read_value<size_type>(payload_position);
        require_record_extent(payload_position, old_count);

        const size_type new_count = static_cast<size_type>(values.size());
        if (new_count != old_count) {
            throw std::length_error(
                "NumericDatabase overwrite requires identical record length");
        }

        write_values(payload_position, values.data(), new_count);
    }

    /** Same-length overwrite for scalar records. */
    void overwrite_scalar(address_type address, value_type value) {
        const value_type values[1]{value};
        overwrite_contiguous(address, values, size_type{1});
    }

    /** Read a record that is required to contain exactly one scalar. */
    bool read_scalar(address_type address, value_type& value) const {
        if (address == address_type{0}) {
            return false;
        }
        ReadGuard guard(lock_);
        size_type position = checked_position(address);
        const size_type count = read_value<size_type>(position);
        if (count != size_type{1}) {
            throw std::runtime_error(
                "NumericDatabase scalar read requested for non-scalar record");
        }
        require_record_extent(position, count);
        read_values(position, &value, size_type{1});
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
    static constexpr void require_value_vector() {
        static_assert(
            std::is_same_v<DataElement<Vector>, value_type>,
            "NumericDatabase vectors must contain value_type and provide data()");
    }

    template <class Vector>
    static constexpr void require_mutable_value_vector() {
        require_value_vector<Vector>();
        static_assert(
            !std::is_const_v<
                std::remove_pointer_t<decltype(std::declval<Vector&>().data())>>,
            "NumericDatabase destination must provide mutable storage");
    }

    [[nodiscard]] address_type push_contiguous(
        const value_type* values,
        size_type count) {
        const size_type units =
            length_unit_count + value_unit_count * count;
        const size_type begin = reserve_units(units);
        if (begin > (std::numeric_limits<size_type>::max)() - units) {
            throw std::overflow_error(
                "NumericDatabase stream position overflow");
        }
        ensure_capacity(begin + units);
        ReadGuard guard(lock_);
        size_type position = begin;
        write_value(position, count);
        write_values(position, values, count);
        return begin + address_type{1};
    }


    void overwrite_contiguous(
        address_type address,
        const value_type* values,
        size_type count) {
        if (address == address_type{0}) {
            throw std::invalid_argument(
                "NumericDatabase cannot overwrite address zero");
        }
        WriteGuard guard(lock_);
        size_type header_position = checked_position(address);
        size_type payload_position = header_position;
        const size_type old_count = read_value<size_type>(payload_position);
        require_record_extent(payload_position, old_count);
        if (count != old_count) {
            throw std::length_error(
                "NumericDatabase overwrite requires identical record length");
        }
        write_values(payload_position, values, count);
    }

    [[nodiscard]] static size_type checked_unit_length(size_type unit_length) {
        if (unit_length == size_type{0}) {
            throw std::invalid_argument(
                "NumericDatabase block length must be greater than zero");
        }
        return unit_length;
    }

    [[nodiscard]] size_type reserve_units(size_type count) noexcept {
        return top_.fetch_add(count, std::memory_order_relaxed);
    }

    void ensure_capacity(size_type required_units) {
        if (capacity_.load(std::memory_order_acquire) >= required_units) {
            return;
        }

        // Only block allocation needs the write lock. The second check avoids growing
        // storage when another writer already supplied enough capacity.
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
                "NumericDatabase address is outside reserved storage");
        }
        return position;
    }

    void require_record_extent(
        size_type payload_position,
        size_type count) const {
        if (count >
            (std::numeric_limits<size_type>::max)() / value_unit_count) {
            throw std::runtime_error(
                "NumericDatabase contains an invalid record length");
        }
        const size_type payload_units = value_unit_count * count;
        const size_type reserved = top_.load(std::memory_order_acquire);
        if (payload_position > reserved ||
            payload_units > reserved - payload_position) {
            throw std::runtime_error(
                "NumericDatabase contains a truncated record");
        }
    }

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

        // Copy chunk-by-chunk because one logical record may span several fixed blocks.
        while (remaining_units > size_type{0}) {
            const size_type block = position / unit_length_;
            const size_type offset = position % unit_length_;
            const size_type copied_units = std::min(
                remaining_units,
                unit_length_ - offset);
            const size_type copied_bytes = copied_units * sizeof(UInt16);
            std::memcpy(
                data_[block].data() + offset,
                source_bytes,
                copied_bytes);
            source_bytes += copied_bytes;
            position += copied_units;
            remaining_units -= copied_units;
        }
    }

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

        // Copy chunk-by-chunk because one logical record may span several fixed blocks.
        while (remaining_units > size_type{0}) {
            const size_type block = position / unit_length_;
            const size_type offset = position % unit_length_;
            const size_type copied_units = std::min(
                remaining_units,
                unit_length_ - offset);
            const size_type copied_bytes = copied_units * sizeof(UInt16);
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
