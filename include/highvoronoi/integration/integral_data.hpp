#pragma once

/**
 * @file integral_data.hpp
 * @brief Optional per-cell measure/integral storage linked to neighbour history.
 *
 * Persistent state is structure-of-arrays, not an array of CellData structs:
 *
 *   neighbour_addresses[cell]                    always present
 *   volumes[cell]                                optional
 *   area_addresses[cell]                         optional
 *   bulk_integral_addresses[cell]                optional
 *   interface_integral_addresses[cell]           optional
 *
 * Disabled arrays remain empty and no records are written for them.
 *
 * CellData is deliberately only a reusable caller-owned I/O buffer.  Reusing
 * the same CellData object makes reads/writes allocation-free once its vector
 * capacities are large enough.
 *
 * Contract:
 *   area()[k] belongs to neighbours()[k]
 *   interface_integral()[k*p + c] belongs to neighbours()[k], component c
 * where p == integral_components().  Neighbour order and multiplicity are
 * therefore semantic and must never be sorted/uniqued independently.
 */

#include <highvoronoi/integration/detail/hybrid_integral_database.hpp>
#include <highvoronoi/core/detail/locks.hpp>
#include <highvoronoi/integration/integral_engine.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

struct IntegralDataOptions {
    bool volume = true;
    bool area = true;
    bool bulk_integral = true;
    bool interface_integral = true;
};

template <class Lock,
          typename IndexT,
          typename AreaScalarT,
          typename IntegralScalarT,
          class NeighbourDatabaseT>
class IntegralData final {
public:
    using LockType = Lock;
    using Index = IndexT;
    using AreaScalar = AreaScalarT;
    using IntegralScalar = IntegralScalarT;
    using Address = std::size_t;
    using NeighbourDatabase = NeighbourDatabaseT;
    using AreaDatabase = detail::HybridAreaDatabase<Lock, AreaScalar, Index>;
    using IntegralDatabase =
        detail::HybridIntegralDatabase<Lock, IntegralScalar, Index>;
    using ReadGuard = detail::ReadLockGuard<Lock>;
    using WriteGuard = detail::WriteLockGuard<Lock>;

    static_assert(
        std::is_integral_v<Index> && std::is_unsigned_v<Index>,
        "IntegralData Index must be an unsigned integral type");
    static_assert(
        std::is_floating_point_v<AreaScalar>,
        "IntegralData AreaScalar must be floating-point");
    static_assert(
        std::is_floating_point_v<IntegralScalar>,
        "IntegralData IntegralScalar must be floating-point");

    /**
     * @brief Reusable I/O buffer for one cell.
     *
     * neighbours() is read-only to callers.  IntegralData fills it from the
     * linked neighbour database and remembers the corresponding neighbour
     * address.  All other fields are mutable work/result buffers.
     */
    class CellData final {
        friend class IntegralData;

    public:
        [[nodiscard]] AreaScalar volume() const noexcept {
            return volume_;
        }

        void set_volume(AreaScalar value) noexcept {
            volume_ = value;
        }

        [[nodiscard]] const std::vector<Index>& neighbours() const noexcept {
            return neighbours_;
        }

        [[nodiscard]] std::vector<AreaScalar>& area() noexcept {
            return area_;
        }

        [[nodiscard]] const std::vector<AreaScalar>& area() const noexcept {
            return area_;
        }

        [[nodiscard]] std::vector<IntegralScalar>& bulk_integral() noexcept {
            return bulk_integral_;
        }

        [[nodiscard]] const std::vector<IntegralScalar>&
        bulk_integral() const noexcept {
            return bulk_integral_;
        }

        [[nodiscard]] std::vector<IntegralScalar>&
        interface_integral() noexcept {
            return interface_integral_;
        }

        [[nodiscard]] const std::vector<IntegralScalar>&
        interface_integral() const noexcept {
            return interface_integral_;
        }

        /** Target neighbour snapshot to which this CellData will be committed. */
        [[nodiscard]] Address neighbour_address() const noexcept {
            return neighbour_address_;
        }

        /** Snapshot from which the current writable values were prepared. */
        [[nodiscard]] Address source_neighbour_address() const noexcept {
            return source_neighbour_address_;
        }

        /** Optional pre-reservation for guaranteed allocation-free reuse. */
        void reserve(
            std::size_t neighbour_capacity,
            std::size_t integral_components) {
            neighbours_.reserve(neighbour_capacity);
            area_.reserve(neighbour_capacity);
            bulk_integral_.reserve(integral_components);
            interface_integral_.reserve(
                checked_product(neighbour_capacity, integral_components));
        }

    private:
        [[nodiscard]] static std::size_t checked_product(
            std::size_t left,
            std::size_t right) {
            if (right != 0 &&
                left > (std::numeric_limits<std::size_t>::max)() / right) {
                throw std::overflow_error("CellData reserve size overflow");
            }
            return left * right;
        }

        void clear_values() noexcept {
            volume_ = AreaScalar{};
            area_.clear();
            bulk_integral_.clear();
            interface_integral_.clear();
        }

        AreaScalar volume_{};
        Address source_neighbour_address_ = Address{0};
        Address neighbour_address_ = Address{0};
        std::vector<Index> neighbours_;
        std::vector<AreaScalar> area_;
        std::vector<IntegralScalar> bulk_integral_;
        std::vector<IntegralScalar> interface_integral_;
    };

    struct EngineRegistration {
        using AreaRegistration = typename AreaDatabase::EngineRegistration;
        using IntegralRegistration = typename IntegralDatabase::EngineRegistration;

        std::optional<AreaRegistration> area;
        std::optional<IntegralRegistration> integral;
    };

    // Persistent storage equivalent of Julia Voronoi_Integral, except neighbours are
    // referenced through immutable neighbour-database addresses instead of copied.
    IntegralData(
        const NeighbourDatabase& neighbour_database,
        std::size_t cell_count,
        std::size_t integral_components,
        IntegralDataOptions options = {},
        std::size_t area_database_unit_length = std::size_t{65536},
        std::size_t integral_database_unit_length = std::size_t{65536})
        : neighbour_database_(&neighbour_database),
          options_(options),
          integral_components_(
              checked_components(integral_components, options)),
          neighbour_addresses_(cell_count, Address{0}) {
        // Allocate only the enabled structure-of-arrays fields. Disabled data has
        // literally no per-cell array and no backing numeric database.
        if (options_.volume) {
            volumes_.resize(cell_count, AreaScalar{});
        }
        if (options_.area) {
            area_addresses_.resize(cell_count, Address{0});
            area_database_ = std::make_unique<AreaDatabase>(
                area_database_unit_length);
        }
        if (options_.bulk_integral) {
            bulk_integral_addresses_.resize(cell_count, Address{0});
        }
        if (options_.interface_integral) {
            interface_integral_addresses_.resize(cell_count, Address{0});
        }
        if (options_.bulk_integral || options_.interface_integral) {
            integral_database_ = std::make_unique<IntegralDatabase>(
                integral_database_unit_length);
        }
    }

    IntegralData(const IntegralData&) = delete;
    IntegralData& operator=(const IntegralData&) = delete;
    IntegralData(IntegralData&&) = delete;
    IntegralData& operator=(IntegralData&&) = delete;

    [[nodiscard]] std::size_t size() const {
        ReadGuard guard(lock_);
        return neighbour_addresses_.size();
    }

    [[nodiscard]] IntegralDataOptions options() const noexcept {
        return options_;
    }

    [[nodiscard]] std::size_t integral_components() const noexcept {
        return integral_components_;
    }

    [[nodiscard]] bool stores_volume() const noexcept {
        return options_.volume;
    }
    [[nodiscard]] bool stores_area() const noexcept {
        return options_.area;
    }
    [[nodiscard]] bool stores_bulk_integral() const noexcept {
        return options_.bulk_integral;
    }
    [[nodiscard]] bool stores_interface_integral() const noexcept {
        return options_.interface_integral;
    }

    /** Test/introspection helpers: disabled arrays are exactly size zero. */
    [[nodiscard]] std::size_t volume_slot_count() const noexcept {
        return volumes_.size();
    }
    [[nodiscard]] std::size_t area_address_slot_count() const noexcept {
        return area_addresses_.size();
    }
    [[nodiscard]] std::size_t bulk_integral_address_slot_count() const noexcept {
        return bulk_integral_addresses_.size();
    }
    [[nodiscard]] std::size_t
    interface_integral_address_slot_count() const noexcept {
        return interface_integral_addresses_.size();
    }

    /** Structural phase-boundary operation. */
    void resize(std::size_t cell_count) {
        WriteGuard guard(lock_);
        neighbour_addresses_.resize(cell_count, Address{0});
        if (options_.volume) {
            volumes_.resize(cell_count, AreaScalar{});
        }
        if (options_.area) {
            area_addresses_.resize(cell_count, Address{0});
        }
        if (options_.bulk_integral) {
            bulk_integral_addresses_.resize(cell_count, Address{0});
        }
        if (options_.interface_integral) {
            interface_integral_addresses_.resize(cell_count, Address{0});
        }
    }

    [[nodiscard]] const NeighbourDatabase& neighbour_database() const noexcept {
        return *neighbour_database_;
    }

    /** Mandatory per-cell link to one immutable neighbour record. */
    void set_neighbour_address(Index cell_index, Address address) {
        WriteGuard guard(lock_);
        require_cell_unlocked(cell_index);
        neighbour_addresses_[static_cast<std::size_t>(cell_index)] = address;
    }

    [[nodiscard]] Address neighbour_address(Index cell_index) const {
        ReadGuard guard(lock_);
        require_cell_unlocked(cell_index);
        return neighbour_addresses_[static_cast<std::size_t>(cell_index)];
    }

    [[nodiscard]] Address area_address(Index cell_index) const {
        if (!options_.area) {
            return Address{0};
        }
        ReadGuard guard(lock_);
        require_cell_unlocked(cell_index);
        return area_addresses_[static_cast<std::size_t>(cell_index)];
    }

    [[nodiscard]] Address bulk_integral_address(Index cell_index) const {
        if (!options_.bulk_integral) {
            return Address{0};
        }
        ReadGuard guard(lock_);
        require_cell_unlocked(cell_index);
        return bulk_integral_addresses_[static_cast<std::size_t>(cell_index)];
    }

    [[nodiscard]] Address interface_integral_address(Index cell_index) const {
        if (!options_.interface_integral) {
            return Address{0};
        }
        ReadGuard guard(lock_);
        require_cell_unlocked(cell_index);
        return interface_integral_addresses_[static_cast<std::size_t>(cell_index)];
    }

    [[nodiscard]] AreaDatabase* area_database() noexcept {
        return area_database_.get();
    }
    [[nodiscard]] const AreaDatabase* area_database() const noexcept {
        return area_database_.get();
    }
    [[nodiscard]] IntegralDatabase* integral_database() noexcept {
        return integral_database_.get();
    }
    [[nodiscard]] const IntegralDatabase* integral_database() const noexcept {
        return integral_database_.get();
    }

    /**
     * Load only the read-only neighbour side and reset all writable buffers.
     * Returns false when no neighbour snapshot has been published for the cell.
     */
    [[nodiscard]] bool prepare_cell(Index cell_index, CellData& data) const {
        data.clear_values();

        Address address = Address{0};
        {
            ReadGuard guard(lock_);
            require_cell_unlocked(cell_index);
            address = neighbour_addresses_[static_cast<std::size_t>(cell_index)];
        }
        data.source_neighbour_address_ = address;
        data.neighbour_address_ = address;
        data.neighbours_.clear();
        if (address == Address{0}) {
            return false;
        }
        return neighbour_database_->read(address, data.neighbours_);
    }

    /**
     * Read one complete enabled cell state into recyclable caller-owned buffers.
     * Disabled fields are returned as zero/empty.  Returns false if any enabled
     * field has not yet been published.
     */
    [[nodiscard]] bool read_cell(Index cell_index, CellData& data) const {
        // Start with the historical neighbour snapshot. Every optional payload read
        // below is interpreted relative to exactly that neighbour ordering.
        bool complete = prepare_cell(cell_index, data);

        Address area_addr = Address{0};
        Address bulk_addr = Address{0};
        Address interface_addr = Address{0};
        {
            ReadGuard guard(lock_);
            require_cell_unlocked(cell_index);
            const std::size_t slot = static_cast<std::size_t>(cell_index);
            if (options_.volume) {
                data.volume_ = volumes_[slot];
            } else {
                data.volume_ = AreaScalar{};
            }
            if (options_.area) {
                area_addr = area_addresses_[slot];
            }
            if (options_.bulk_integral) {
                bulk_addr = bulk_integral_addresses_[slot];
            }
            if (options_.interface_integral) {
                interface_addr = interface_integral_addresses_[slot];
            }
        }

        if (options_.area) {
            if (area_addr == Address{0} ||
                !area_database_->read(area_addr, data.area_)) {
                data.area_.clear();
                complete = false;
            }
        } else {
            data.area_.clear();
        }

        if (options_.bulk_integral) {
            if (bulk_addr == Address{0} ||
                !integral_database_->read(bulk_addr, data.bulk_integral_)) {
                data.bulk_integral_.clear();
                complete = false;
            }
        } else {
            data.bulk_integral_.clear();
        }

        if (options_.interface_integral) {
            if (interface_addr == Address{0} ||
                !integral_database_->read(
                    interface_addr,
                    data.interface_integral_)) {
                data.interface_integral_.clear();
                complete = false;
            }
        } else {
            data.interface_integral_.clear();
        }

        return complete && validate_buffer_layout(data);
    }

    /**
     * Read only the geometry payload required by geometry-consuming algorithms.
     *
     * Unlike read_cell(), this intentionally ignores enabled bulk/interface
     * integral fields. A geometry source is therefore considered usable when
     * its neighbour snapshot, volume and area record are complete even if its
     * function-integral payloads are absent or irrelevant to the consumer.
     */
    [[nodiscard]] bool read_geometry_cell(
        Index cell_index,
        CellData& data) const {
        bool complete = prepare_cell(cell_index, data);

        if (!options_.volume || !options_.area) {
            return false;
        }

        Address area_addr = Address{0};
        {
            ReadGuard guard(lock_);
            require_cell_unlocked(cell_index);
            const std::size_t slot = static_cast<std::size_t>(cell_index);
            data.volume_ = volumes_[slot];
            area_addr = area_addresses_[slot];
        }

        if (area_addr == Address{0} ||
            !area_database_->read(area_addr, data.area_)) {
            data.area_.clear();
            complete = false;
        }

        // Geometry consumers must not accidentally observe stale function data
        // left in a recycled CellData buffer.
        data.bulk_integral_.clear();
        data.interface_integral_.clear();

        return complete &&
            data.area_.size() == data.neighbours_.size();
    }

    /**
     * Retarget one prepared CellData object to a new immutable neighbour record.
     *
     * Existing area/interface entries are transferred occurrence-by-occurrence
     * for neighbours that survive in the new sorted list.  Duplicate neighbour
     * entries are therefore matched one-to-one in ordinal order.  Newly added
     * neighbour occurrences receive zero data.  `matched[k]` reports whether
     * new neighbour occurrence `k` had an old counterpart.
     *
     * The source neighbour address remembered by CellData is deliberately not
     * changed.  A later write_cell() therefore commits old -> new atomically and
     * rejects the write if another update has meanwhile replaced the source
     * snapshot.
     *
     * Scratch vectors are caller-owned so repeated update preparation can be
     * allocation-free after capacity has been reserved once.
     */
    void retarget_cell(
        CellData& data,
        Address target_neighbour_address,
        const std::vector<Index>& new_neighbours,
        std::vector<std::uint8_t>& matched,
        std::vector<AreaScalar>& area_scratch,
        std::vector<IntegralScalar>& interface_scratch) const {
        if (target_neighbour_address == Address{0}) {
            throw std::invalid_argument(
                "IntegralData cannot retarget to neighbour address zero");
        }
        if (!std::is_sorted(new_neighbours.begin(), new_neighbours.end())) {
            throw std::invalid_argument(
                "IntegralData neighbour records must be sorted");
        }
        if (!std::is_sorted(data.neighbours_.begin(), data.neighbours_.end())) {
            throw std::logic_error(
                "IntegralData stored neighbour snapshot is not sorted");
        }

        const std::size_t new_count = new_neighbours.size();
        matched.assign(new_count, std::uint8_t{0});

        if (options_.area) {
            area_scratch.assign(new_count, AreaScalar{});
        } else {
            area_scratch.clear();
        }
        if (options_.interface_integral) {
            interface_scratch.assign(
                checked_interface_size(new_count, integral_components_),
                IntegralScalar{});
        } else {
            interface_scratch.clear();
        }

        // Julia analogue: set_neighbors(...). Merge the old and new sorted neighbour
        // lists occurrence-by-occurrence, carrying aligned area/interface values along.
        // Duplicate neighbours are therefore preserved and matched independently.
        std::size_t old_position = 0;
        std::size_t new_position = 0;
        while (old_position < data.neighbours_.size() &&
               new_position < new_neighbours.size()) {
            const Index old_neighbour = data.neighbours_[old_position];
            const Index new_neighbour = new_neighbours[new_position];

            if (old_neighbour < new_neighbour) {
                ++old_position;
                continue;
            }
            if (new_neighbour < old_neighbour) {
                ++new_position;
                continue;
            }

            matched[new_position] = std::uint8_t{1};
            if (options_.area && old_position < data.area_.size()) {
                area_scratch[new_position] = data.area_[old_position];
            }
            if (options_.interface_integral) {
                const std::size_t old_begin =
                    checked_interface_size(old_position, integral_components_);
                const std::size_t new_begin =
                    checked_interface_size(new_position, integral_components_);
                if (old_begin + integral_components_ <=
                    data.interface_integral_.size()) {
                    std::copy_n(
                        data.interface_integral_.begin() +
                            static_cast<std::ptrdiff_t>(old_begin),
                        integral_components_,
                        interface_scratch.begin() +
                            static_cast<std::ptrdiff_t>(new_begin));
                }
            }
            ++old_position;
            ++new_position;
        }

        // From here on CellData represents the target snapshot; source_neighbour_address_
        // deliberately still identifies the snapshot from which this transaction began.
        data.neighbours_.assign(new_neighbours.begin(), new_neighbours.end());
        data.neighbour_address_ = target_neighbour_address;

        if (options_.area) {
            data.area_.swap(area_scratch);
            area_scratch.clear();
        } else {
            data.area_.clear();
        }
        if (options_.bulk_integral) {
            if (data.bulk_integral_.size() != integral_components_) {
                data.bulk_integral_.assign(
                    integral_components_, IntegralScalar{});
            }
        } else {
            data.bulk_integral_.clear();
        }
        if (options_.interface_integral) {
            data.interface_integral_.swap(interface_scratch);
            interface_scratch.clear();
        } else {
            data.interface_integral_.clear();
        }
    }

    /**
     * Store enabled writable fields from CellData and atomically publish both
     * their addresses and the target neighbour snapshot.
     *
     * neighbours() is strictly read-only to the caller.  A CellData object may
     * either still target its original snapshot or have been retargeted through
     * retarget_cell().  In both cases the remembered source address is checked
     * before and after append-only database writes.  Thus a failed/stale update
     * cannot silently attach geometry data to the wrong neighbour ordering.
     *
     * If a flag is false, the corresponding CellData content is ignored even if
     * the caller filled it manually.
     */
    void write_cell(Index cell_index, const CellData& data) {
        Address current_neighbour_address = Address{0};
        {
            ReadGuard guard(lock_);
            require_cell_unlocked(cell_index);
            current_neighbour_address =
                neighbour_addresses_[static_cast<std::size_t>(cell_index)];
        }

        if (data.source_neighbour_address_ != current_neighbour_address) {
            throw std::logic_error(
                "IntegralData CellData belongs to a stale source neighbour snapshot");
        }

        if ((options_.area || options_.interface_integral) &&
            data.neighbour_address_ == Address{0}) {
            throw std::logic_error(
                "Area/interface data require a target neighbour record");
        }

        validate_writable_layout(data);

        Address new_area = Address{0};
        Address new_bulk = Address{0};
        Address new_interface = Address{0};

        // Append new immutable payload records first. Publication of their per-cell
        // addresses happens only after the source neighbour snapshot is revalidated.
        if (options_.area) {
            new_area = area_database_->push(data.area_);
        }
        if (options_.bulk_integral) {
            new_bulk = integral_database_->push(data.bulk_integral_);
        }
        if (options_.interface_integral) {
            new_interface = integral_database_->push(data.interface_integral_);
        }

        WriteGuard guard(lock_);
        require_cell_unlocked(cell_index);
        const std::size_t slot = static_cast<std::size_t>(cell_index);

        // Revalidate after database writes before publication. Append-only
        // records written before a failed revalidation are harmless history.
        if (neighbour_addresses_[slot] != data.source_neighbour_address_) {
            throw std::logic_error(
                "IntegralData source neighbour snapshot changed during cell write");
        }

        // Atomic publication point for this cell: all current per-cell addresses now
        // refer to payloads aligned with the same target neighbour snapshot.
        neighbour_addresses_[slot] = data.neighbour_address_;
        if (options_.volume) {
            volumes_[slot] = data.volume_;
        }
        if (options_.area) {
            area_addresses_[slot] = new_area;
        }
        if (options_.bulk_integral) {
            bulk_integral_addresses_[slot] = new_bulk;
        }
        if (options_.interface_integral) {
            interface_integral_addresses_[slot] = new_interface;
        }
    }



    /**
     * @brief Overwrite an existing area record without changing its length.
     *
     * This is intended for serial cleanup of freshly produced records. The
     * underlying database verifies the stored length before modifying payload.
     */
    template <class ValueVector>
    void overwrite_area_record(Address address, const ValueVector& values) {
        if (!options_.area) {
            throw std::logic_error("IntegralData area storage is disabled");
        }
        area_database_->overwrite(address, values);
    }

    /**
     * @brief Overwrite an existing integral record without changing its length.
     *
     * Bulk and interface integrals share the same IntegralDatabase; the caller
     * chooses the address to overwrite. Length changes are rejected by the DB.
     */
    template <class ValueVector>
    void overwrite_integral_record(Address address, const ValueVector& values) {
        if (!options_.bulk_integral && !options_.interface_integral) {
            throw std::logic_error("IntegralData integral storage is disabled");
        }
        integral_database_->overwrite(address, values);
    }

    /** Verify all enabled layout invariants for one published cell. */
    [[nodiscard]] bool validate_layout(Index cell_index) const {
        CellData scratch;
        return read_cell(cell_index, scratch);
    }

    /**
     * Register one coherent engine.  IntegralData creates exactly one adapter
     * object and gives typed views of that same adapter to the two hybrid DBs.
     */
    template <class EngineT>
    EngineRegistration register_engine(
        const std::shared_ptr<EngineT>& engine) {
        static_assert(
            std::is_base_of_v<
                IntegralEngine<
                    typename EngineT::MeshEngine,
                    AreaScalar,
                    IntegralScalar>,
                EngineT>,
            "EngineT must derive from the matching IntegralEngine base");

        if (!engine) {
            throw std::invalid_argument(
                "IntegralData requires a non-null integral engine");
        }
        if ((options_.bulk_integral || options_.interface_integral) &&
            engine->integral_components() != integral_components_) {
            throw std::invalid_argument(
                "Integral engine component count does not match IntegralData");
        }

        using Adapter = detail::IntegralEngineDatabaseAdapter<EngineT>;
        auto adapter = std::make_shared<Adapter>(engine);

        EngineRegistration result;
        if (options_.area) {
            std::shared_ptr<const detail::AreaEngineInterface<AreaScalar, Index>>
                area_adapter = adapter;
            result.area = area_database_->register_engine(
                std::move(area_adapter));
        }
        if (options_.bulk_integral || options_.interface_integral) {
            std::shared_ptr<
                const detail::IntegralEngineInterface<IntegralScalar, Index>>
                integral_adapter = adapter;
            result.integral = integral_database_->register_engine(
                std::move(integral_adapter));
        }
        return result;
    }

    /**
     * Publish one engine-local cell into the structure-of-arrays state.
     * Only enabled fields are touched.
     */
    template <class EngineT>
    void publish_engine_cell(
        Index target_cell,
        const EngineT& engine,
        const EngineRegistration& registration,
        Index local_cell,
        Address neighbour_address_value) {
        AreaScalar volume_value{};
        if (options_.volume &&
            !engine.read_volume(local_cell, volume_value)) {
            throw std::logic_error(
                "Integral engine did not provide requested cell volume");
        }

        Address area_value = Address{0};
        if (options_.area) {
            if (!registration.area) {
                throw std::logic_error("Missing area engine registration");
            }
            if (const auto local = engine.area_address(local_cell)) {
                area_value = registration.area->logical_address(*local);
            }
        }

        Address bulk_value = Address{0};
        Address interface_value = Address{0};
        if (options_.bulk_integral || options_.interface_integral) {
            if (!registration.integral) {
                throw std::logic_error("Missing integral engine registration");
            }
            if (options_.bulk_integral) {
                if (const auto local = engine.bulk_integral_address(local_cell)) {
                    bulk_value = registration.integral->logical_address(*local);
                }
            }
            if (options_.interface_integral) {
                if (const auto local =
                        engine.interface_integral_address(local_cell)) {
                    interface_value =
                        registration.integral->logical_address(*local);
                }
            }
        }

        WriteGuard guard(lock_);
        require_cell_unlocked(target_cell);
        const std::size_t slot = static_cast<std::size_t>(target_cell);
        neighbour_addresses_[slot] = neighbour_address_value;
        if (options_.volume) {
            volumes_[slot] = volume_value;
        }
        if (options_.area) {
            area_addresses_[slot] = area_value;
        }
        if (options_.bulk_integral) {
            bulk_integral_addresses_[slot] = bulk_value;
        }
        if (options_.interface_integral) {
            interface_integral_addresses_[slot] = interface_value;
        }
    }

private:
    [[nodiscard]] static std::size_t checked_components(
        std::size_t value,
        const IntegralDataOptions& options) {
        if ((options.bulk_integral || options.interface_integral) && value == 0) {
            throw std::invalid_argument(
                "Enabled integral storage requires at least one component");
        }
        return value;
    }

    [[nodiscard]] static std::size_t checked_interface_size(
        std::size_t neighbour_count,
        std::size_t components) {
        if (components != 0 &&
            neighbour_count >
                (std::numeric_limits<std::size_t>::max)() / components) {
            throw std::overflow_error(
                "Interface integral layout size overflow");
        }
        return neighbour_count * components;
    }

    void validate_writable_layout(const CellData& data) const {
        const std::size_t neighbour_count = data.neighbours_.size();
        if (options_.area && data.area_.size() != neighbour_count) {
            throw std::invalid_argument(
                "Area record must contain exactly one value per neighbour occurrence");
        }
        if (options_.bulk_integral &&
            data.bulk_integral_.size() != integral_components_) {
            throw std::invalid_argument(
                "Bulk integral record has wrong component count");
        }
        if (options_.interface_integral &&
            data.interface_integral_.size() !=
                checked_interface_size(
                    neighbour_count,
                    integral_components_)) {
            throw std::invalid_argument(
                "Interface integral record must be neighbour-major and aligned with neighbours");
        }
    }

    [[nodiscard]] bool validate_buffer_layout(const CellData& data) const {
        const std::size_t neighbour_count = data.neighbours_.size();
        if (options_.area && data.area_.size() != neighbour_count) {
            return false;
        }
        if (options_.bulk_integral &&
            data.bulk_integral_.size() != integral_components_) {
            return false;
        }
        if (options_.interface_integral &&
            data.interface_integral_.size() !=
                checked_interface_size(
                    neighbour_count,
                    integral_components_)) {
            return false;
        }
        return true;
    }

    void require_cell_unlocked(Index cell_index) const {
        if (static_cast<std::size_t>(cell_index) >= neighbour_addresses_.size()) {
            throw std::out_of_range("IntegralData cell index out of range");
        }
    }

    const NeighbourDatabase* neighbour_database_;
    const IntegralDataOptions options_;
    const std::size_t integral_components_;

    // Structure-of-arrays cell state. Disabled optional vectors stay empty.
    std::vector<Address> neighbour_addresses_;
    std::vector<AreaScalar> volumes_;
    std::vector<Address> area_addresses_;
    std::vector<Address> bulk_integral_addresses_;
    std::vector<Address> interface_integral_addresses_;

    // Databases themselves are not even constructed when their data category
    // is completely disabled.
    std::unique_ptr<AreaDatabase> area_database_;
    std::unique_ptr<IntegralDatabase> integral_database_;

    mutable Lock lock_{};
};

} // namespace highvoronoi
