#pragma once

/**
 * @file hybrid_numeric_database.hpp
 * @brief Logical-address router for stored and computed numeric records.
 *
 * Area and integral databases use this same routing primitive but remain
 * semantically distinct types through different engine-access policies.
 * Registered engine records must be immutable/reproducible for as long as
 * their virtual logical addresses remain visible.
 */

#include <highvoronoi/core/detail/locks.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace highvoronoi::detail {

template <class Lock,
          class ValueT,
          class StoredDatabaseT,
          class EngineInterfaceT,
          class EngineAccessT>
class HybridNumericDatabase {
public:
    using LockType = Lock;
    using value_type = ValueT;
    using StoredDatabase = StoredDatabaseT;
    using EngineInterface = EngineInterfaceT;
    using EngineAccess = EngineAccessT;
    using Address = std::size_t;
    using address_type = Address;
    using ReadGuard = ReadLockGuard<Lock>;
    using WriteGuard = WriteLockGuard<Lock>;

    struct EngineRegistration {
        std::size_t engine_index = 0;
        Address first_address = 0;
        Address address_capacity = 0;

        [[nodiscard]] Address end_address() const noexcept {
            return first_address + address_capacity;
        }

        [[nodiscard]] Address logical_address(Address local) const {
            if (local >= address_capacity) {
                throw std::out_of_range(
                    "Hybrid numeric engine-local address out of range");
            }
            return first_address + local;
        }
    };

    explicit HybridNumericDatabase(
        std::size_t stored_unit_length = std::size_t{65536})
        : stored_(stored_unit_length) {}

    HybridNumericDatabase(const HybridNumericDatabase&) = delete;
    HybridNumericDatabase& operator=(const HybridNumericDatabase&) = delete;
    HybridNumericDatabase(HybridNumericDatabase&&) = delete;
    HybridNumericDatabase& operator=(HybridNumericDatabase&&) = delete;

    template <class ValueVector>
    Address push(const ValueVector& values) {
        // First append to the physical stored database, then expose that immutable
        // record through the common logical Stored/Engine address space.
        const Address physical = stored_.push(values);
        return append_stored_record(physical);
    }

    Address push_scalar(value_type value) {
        const Address physical = stored_.push_scalar(value);
        return append_stored_record(physical);
    }

    template <class ValueVector>
    bool read(Address address, ValueVector& values) const {
        values.clear();
        if (address == Address{0}) {
            return false;
        }
        // Logical addresses are intentionally source-agnostic. Resolve once, then
        // dispatch either to stored history or to the registered computed engine.
        const ResolvedAddress resolved = resolve_address(address);
        if (resolved.kind == SegmentKind::Stored) {
            return stored_.read(resolved.physical_address, values);
        }
        return EngineAccess::read(
            *resolved.engine,
            resolved.local_address,
            values);
    }

    bool read_scalar(Address address, value_type& value) const {
        if (address == Address{0}) {
            return false;
        }
        // Logical addresses are intentionally source-agnostic. Resolve once, then
        // dispatch either to stored history or to the registered computed engine.
        const ResolvedAddress resolved = resolve_address(address);
        if (resolved.kind == SegmentKind::Stored) {
            return stored_.read_scalar(resolved.physical_address, value);
        }
        std::vector<value_type> scratch;
        if (!EngineAccess::read(
                *resolved.engine,
                resolved.local_address,
                scratch)) {
            return false;
        }
        if (scratch.size() != 1) {
            throw std::runtime_error(
                "Hybrid numeric scalar read requested for non-scalar engine record");
        }
        value = scratch.front();
        return true;
    }


    /**
     * @brief Overwrite a stored logical record without changing its length.
     *
     * Engine-backed virtual records remain immutable. Cleanup code must
     * materialize such data into a fresh stored record before it can overwrite
     * the payload.
     */
    template <class ValueVector>
    void overwrite(Address address, const ValueVector& values) {
        if (address == Address{0}) {
            throw std::invalid_argument(
                "Hybrid numeric database cannot overwrite address zero");
        }
        const ResolvedAddress resolved = resolve_address(address);
        if (resolved.kind != SegmentKind::Stored) {
            throw std::logic_error(
                "Hybrid numeric database cannot overwrite engine records");
        }
        stored_.overwrite(resolved.physical_address, values);
    }

    void overwrite_scalar(Address address, value_type value) {
        if (address == Address{0}) {
            throw std::invalid_argument(
                "Hybrid numeric database cannot overwrite address zero");
        }
        const ResolvedAddress resolved = resolve_address(address);
        if (resolved.kind != SegmentKind::Stored) {
            throw std::logic_error(
                "Hybrid numeric database cannot overwrite engine records");
        }
        stored_.overwrite_scalar(resolved.physical_address, value);
    }

    EngineRegistration register_engine(
        std::shared_ptr<const EngineInterface> engine) {
        if (!engine) {
            throw std::invalid_argument(
                "Hybrid numeric database requires a non-null engine");
        }

        // Register one immutable virtual address interval for this engine. Later
        // stored records continue after that interval in the same logical space.
        const Address capacity = EngineAccess::capacity(*engine);
        WriteGuard guard(address_lock_);
        const std::size_t engine_index = engines_.size();
        engines_.push_back(std::move(engine));

        if (capacity == Address{0}) {
            return EngineRegistration{engine_index, Address{0}, Address{0}};
        }
        if (next_address_ >
            (std::numeric_limits<Address>::max)() - capacity) {
            throw std::overflow_error(
                "Hybrid numeric logical address overflow");
        }

        const Address first = next_address_;
        const Address end = first + capacity;
        next_address_ = end;
        segments_.push_back(Segment{
            first,
            end,
            SegmentKind::Engine,
            engine_index});
        return EngineRegistration{engine_index, first, capacity};
    }

    [[nodiscard]] const StoredDatabase& stored_database() const noexcept {
        return stored_;
    }

private:
    enum class SegmentKind : unsigned char { Stored, Engine };

    struct Segment {
        Address begin = 0;
        Address end = 0;
        SegmentKind kind = SegmentKind::Stored;
        std::size_t source_begin = 0;

        [[nodiscard]] bool contains(Address address) const noexcept {
            return begin <= address && address < end;
        }
    };

    struct ResolvedAddress {
        SegmentKind kind = SegmentKind::Stored;
        Address physical_address = 0;
        std::shared_ptr<const EngineInterface> engine;
        Address local_address = 0;
    };

    [[nodiscard]] Address append_stored_record(Address physical_address) {
        WriteGuard guard(address_lock_);
        const std::size_t slot = stored_physical_addresses_.size();
        stored_physical_addresses_.push_back(physical_address);

        if (next_address_ == (std::numeric_limits<Address>::max)()) {
            throw std::overflow_error(
                "Hybrid numeric logical address overflow");
        }
        const Address logical = next_address_++;

        // Consecutive stored records are coalesced into one routing segment. This
        // keeps address lookup compact without changing individual logical addresses.
        if (!segments_.empty()) {
            Segment& last = segments_.back();
            const std::size_t last_count =
                static_cast<std::size_t>(last.end - last.begin);
            if (last.kind == SegmentKind::Stored &&
                last.end == logical &&
                last.source_begin + last_count == slot) {
                ++last.end;
                return logical;
            }
        }

        segments_.push_back(Segment{
            logical,
            logical + Address{1},
            SegmentKind::Stored,
            slot});
        return logical;
    }

    [[nodiscard]] ResolvedAddress resolve_address(Address address) const {
        ReadGuard guard(address_lock_);
        const Segment segment = locate_segment_unlocked(address);
        const Address offset = address - segment.begin;

        if (segment.kind == SegmentKind::Stored) {
            const std::size_t slot =
                segment.source_begin + static_cast<std::size_t>(offset);
            return ResolvedAddress{
                SegmentKind::Stored,
                stored_physical_addresses_.at(slot),
                {},
                0};
        }

        return ResolvedAddress{
            SegmentKind::Engine,
            0,
            engines_.at(segment.source_begin),
            offset};
    }

    [[nodiscard]] Segment locate_segment_unlocked(Address address) const {
        if (address == Address{0} || segments_.empty()) {
            throw std::out_of_range(
                "Unknown hybrid numeric database address");
        }
        const auto upper = std::upper_bound(
            segments_.begin(),
            segments_.end(),
            address,
            [](Address value, const Segment& segment) {
                return value < segment.begin;
            });
        if (upper == segments_.begin()) {
            throw std::out_of_range(
                "Unknown hybrid numeric database address");
        }
        const Segment segment = *std::prev(upper);
        if (!segment.contains(address)) {
            throw std::out_of_range(
                "Unknown hybrid numeric database address");
        }
        return segment;
    }

    StoredDatabase stored_;
    std::vector<std::shared_ptr<const EngineInterface>> engines_;
    std::vector<Segment> segments_;
    std::vector<Address> stored_physical_addresses_;
    Address next_address_ = Address{1};
    mutable Lock address_lock_{};
};

} // namespace highvoronoi::detail
