#pragma once

/**
 * @file neighbour_storage.hpp
 * @brief Current neighbour-record address plus dirty state per stable cell.
 *
 * The storage deliberately contains only three pieces of state:
 *
 * - one neighbour database shared by the complete mesh,
 * - one global address vector with exactly one current logical address per
 *   stable internal cell,
 * - one dirty bit per stable internal cell.
 *
 * The database is append-only. Replacing the current address of a cell does not
 * destroy older records; callers that retained an old logical address may keep
 * reading that historical neighbour list directly from the same database.
 *
 * Consumer dirty trackers and the global neighbour version are mesh-level
 * concerns and therefore intentionally do not live in this storage class.
 */

#include <highvoronoi/core/detail/atomic_bit_vector.hpp>
#include <highvoronoi/storage/neighbour/neighbour_database.hpp>
#include <highvoronoi/storage/read_write_list.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace highvoronoi::detail {

template <class Lock,
          class IndexT,
          class NeighbourDatabaseT = NeighbourDatabase<Lock, IndexT>>
class NeighbourStorage final {
public:
    using LockType = Lock;
    using Index = IndexT;
    using Address = std::size_t;
    using Database = NeighbourDatabaseT;
    using AddressVector =
        ReadWriteAddressList<std::vector<Address>, Lock>;
    using DirtyVector = NeighbourDirtyVector<Lock>;

    static_assert(std::is_same_v<typename Database::Index, Index>,
                  "NeighbourStorage database Index must match storage Index.");

    explicit NeighbourStorage(
        std::size_t cell_count = 0,
        std::size_t database_unit_length = std::size_t{65536},
        bool initially_dirty = true)
        : database_(database_unit_length),
          dirty_(0) {
        addresses_.resize(cell_count, Address{0});
        dirty_.resize(cell_count, false);
        if (initially_dirty) {
            set_dirty_range(0, cell_count);
        }
    }

    NeighbourStorage(const NeighbourStorage&) = delete;
    NeighbourStorage& operator=(const NeighbourStorage&) = delete;
    NeighbourStorage(NeighbourStorage&&) = delete;
    NeighbourStorage& operator=(NeighbourStorage&&) = delete;

    [[nodiscard]] std::size_t size() const noexcept {
        return addresses_.size();
    }

    /** Structural phase-boundary operation; appended cells start dirty. */
    void resize(std::size_t cell_count) {
        const std::size_t old_size = size();
        addresses_.resize(cell_count, Address{0});
        dirty_.resize(cell_count, false);
        if (cell_count > old_size) {
            set_dirty_range(old_size, cell_count);
        }
    }

    /** Read the currently published record, even when this cell is dirty. */
    template <class IndexVector>
    [[nodiscard]] bool load(Index cell, IndexVector& buffer) const {
        return database_.read(address(cell), buffer);
    }

    /** Read an explicitly retained historical logical address. */
    template <class IndexVector>
    [[nodiscard]] bool read_address(
        Address logical_address,
        IndexVector& buffer) const {
        return database_.read(logical_address, buffer);
    }

    /** Return the one current logical address of this stable internal cell. */
    [[nodiscard]] Address address(Index cell) const {
        return addresses_[checked_cell(cell)];
    }

    /** Publish an already existing logical database address. Dirty is unchanged. */
    void publish_address(Index cell, Address logical_address) {
        addresses_.set(checked_cell(cell), logical_address);
    }

    /**
     * Append one immutable stored neighbour record and publish its address.
     * Dirty is deliberately unchanged; the mesh clears it only after a
     * successful explicit compute_neighbors() operation.
     */
    template <class IndexVector>
    Address store(
        Index cell,
        const IndexVector& neighbours) {
        const std::size_t c = checked_cell(cell);
        if (!std::is_sorted(neighbours.begin(), neighbours.end())) {
            throw std::invalid_argument(
                "Internal neighbour lists must be sorted before storage.");
        }
        const Address logical_address = database_.push(neighbours);
        addresses_.set(c, logical_address);
        return logical_address;
    }

    [[nodiscard]] bool dirty(Index cell) const {
        return dirty_.test(checked_cell(cell));
    }

    void set_dirty(Index cell, bool value = true) {
        const std::size_t c = checked_cell(cell);
        if (value) {
            dirty_.set(c);
        } else {
            dirty_.reset(c);
        }
    }

    [[nodiscard]] bool has_record(Index cell) const {
        return address(cell) != Address{0};
    }

    /** Expose mesh-owned dirty bits for word-wise propagation at mesh level. */
    [[nodiscard]] const DirtyVector& dirty_vector() const noexcept {
        return dirty_;
    }

    [[nodiscard]] Database& database() noexcept {
        return database_;
    }

    [[nodiscard]] const Database& database() const noexcept {
        return database_;
    }

private:
    [[nodiscard]] std::size_t checked_cell(Index cell) const {
        const std::size_t c = static_cast<std::size_t>(cell);
        if (c >= size()) {
            throw std::out_of_range(
                "NeighbourStorage stable internal cell index out of range.");
        }
        return c;
    }

    void set_dirty_range(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            dirty_.set(i);
        }
    }

    Database database_;
    AddressVector addresses_;
    DirtyVector dirty_;
};

} // namespace highvoronoi::detail
