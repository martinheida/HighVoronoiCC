#pragma once

/**
 * @file hybrid_neighbour_database.hpp
 * @brief Unified logical address space for stored and computed neighbour lists.
 *
 * HybridNeighbourDatabase mirrors the address-routing part of HybridDataBase.
 * One logical, one-based address space contains both append-only stored
 * neighbour records and immutable virtual records supplied by compute engines.
 * Address zero is reserved for "no neighbour record".
 *
 * Stored records live in one NeighbourDatabase backend. A registered engine
 * reserves one virtual address per engine-local ordinary cell. Reading such an
 * address delegates to Engine::read_neighbours(local_cell, sigma) and translates
 * engine-local ordinary neighbour indices by the node offset supplied at
 * registration. High-end boundary-mirror encodings are passed through unchanged.
 *
 * Engine-backed neighbour records are required to be immutable in the logical
 * sense: once an engine is registered, reading one of its virtual addresses must
 * keep returning the neighbour list represented by that address. Engines whose
 * neighbour authority can disappear must not register that mutable state as a
 * persistent virtual record; the host must materialize it into stored records
 * before the engine state changes.
 *
 * Registration is a structural operation and requires external synchronization.
 * Ordinary stored pushes/reads retain the lock behavior of NeighbourDatabase.
 */

#include <highvoronoi/storage/neighbour/neighbour_database.hpp>
#include <highvoronoi/core/detail/locks.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi::detail {

template <class Lock,
          class IndexT,
          class StoreNeighbourDatabaseT = NeighbourDatabase<Lock, IndexT>>
class HybridNeighbourDatabase final {
public:
    using LockType = Lock;
    using Index = IndexT;
    using Address = std::size_t;
    using Sigma = std::vector<Index>;
    using StoreDatabaseType = StoreNeighbourDatabaseT;
    using StoredDatabase = StoreDatabaseType;
    using ReadGuard = ReadLockGuard<LockType>;
    using WriteGuard = WriteLockGuard<LockType>;

    static_assert(std::is_same_v<typename StoreDatabaseType::Index, Index>,
                  "Stored neighbour database Index must match HybridNeighbourDatabase Index.");

    struct EngineRegistration {
        std::size_t engine_index = 0;
        Address first_address = 0;
        Address address_capacity = 0;
        Index node_offset = Index{0};

        [[nodiscard]] Address end_address() const noexcept {
            return first_address + address_capacity;
        }
    };

private:
    enum class SegmentKind : unsigned char {
        Stored,
        Engine
    };

    class EngineConcept {
    public:
        virtual ~EngineConcept() = default;
        [[nodiscard]] virtual Index node_count() const noexcept = 0;
        virtual void read_neighbours(Index local_cell, Sigma& neighbours) const = 0;
    };

    template <class EngineT>
    class EngineModel final : public EngineConcept {
    public:
        using Engine = EngineT;

        explicit EngineModel(std::shared_ptr<Engine> engine)
            : engine_(std::move(engine)) {}

        [[nodiscard]] Index node_count() const noexcept override {
            return engine_->node_count();
        }

        void read_neighbours(Index local_cell, Sigma& neighbours) const override {
            neighbours.clear();
            if (!engine_->read_neighbours(local_cell, neighbours)) {
                throw std::logic_error(
                    "Registered neighbour engine no longer provides its immutable virtual record.");
            }
        }

    private:
        std::shared_ptr<Engine> engine_;
    };

    struct Segment {
        Address begin = 0; // inclusive, one-based logical address
        Address end = 0;   // exclusive
        SegmentKind kind = SegmentKind::Stored;
        std::size_t source_begin = 0; // stored-address slot or engine index
        Index node_offset = Index{0};

        [[nodiscard]] bool contains(Address address) const noexcept {
            return address >= begin && address < end;
        }
    };

public:
    explicit HybridNeighbourDatabase(
        std::size_t unit_length = std::size_t{65536})
        : stored_(unit_length) {}

    HybridNeighbourDatabase(const HybridNeighbourDatabase&) = delete;
    HybridNeighbourDatabase& operator=(const HybridNeighbourDatabase&) = delete;
    HybridNeighbourDatabase(HybridNeighbourDatabase&&) = delete;
    HybridNeighbourDatabase& operator=(HybridNeighbourDatabase&&) = delete;

    [[nodiscard]] StoreDatabaseType& stored_database() noexcept {
        return stored_;
    }

    [[nodiscard]] const StoreDatabaseType& stored_database() const noexcept {
        return stored_;
    }

    [[nodiscard]] Address next_logical_address() const {
        ReadGuard guard(address_lock_);
        return next_address_;
    }

    [[nodiscard]] std::size_t engine_count() const {
        ReadGuard guard(address_lock_);
        return engines_.size();
    }

    /**
     * @brief Reserve one immutable virtual neighbour record per engine-local cell.
     *
     * The engine must declare neighbour support for its complete local node range.
     * Local ordinary neighbour indices are translated by node_offset on reads.
     */
    template <class EngineT>
    [[nodiscard]] EngineRegistration register_engine(
        std::shared_ptr<EngineT> engine,
        Index node_offset) {
        static_assert(std::is_same_v<typename EngineT::Index, Index>,
                      "Engine Index must match HybridNeighbourDatabase Index.");
        static_assert(std::is_same_v<typename EngineT::Sigma, Sigma>,
                      "Engine Sigma must be std::vector<Index> with matching Index.");

        if (!engine) {
            throw std::invalid_argument(
                "HybridNeighbourDatabase requires a non-null engine.");
        }
        if (!engine->provides_neighbours()) {
            throw std::invalid_argument(
                "HybridNeighbourDatabase can register only engines with authoritative neighbour support.");
        }
        validate_node_offset(*engine, node_offset);

        const Address capacity = static_cast<Address>(engine->node_count());
        std::shared_ptr<EngineConcept> model =
            std::make_shared<EngineModel<EngineT>>(engine);

        WriteGuard guard(address_lock_);
        if (capacity > (std::numeric_limits<Address>::max)() - next_address_) {
            throw std::overflow_error(
                "HybridNeighbourDatabase logical address overflow.");
        }

        const std::size_t engine_index = engines_.size();
        const Address first = next_address_;
        engines_.push_back(std::move(model));
        if (capacity != 0) {
            segments_.push_back(Segment{
                first,
                first + capacity,
                SegmentKind::Engine,
                engine_index,
                node_offset});
            next_address_ += capacity;
        }

        return EngineRegistration{
            engine_index,
            first,
            capacity,
            node_offset};
    }

    /** Append one stored neighbour list and return its logical address. */
    template <class IndexVector>
    [[nodiscard]] Address push(const IndexVector& neighbours) {
        const Address physical = stored_.push(neighbours);
        return append_stored_record(physical);
    }

    /**
     * Read either a stored or virtual engine-backed neighbour record.
     * Address zero retains the ordinary NeighbourDatabase "no record" contract.
     */
    template <class IndexVector>
    [[nodiscard]] bool read(
        Address address,
        IndexVector& neighbours) const {
        static_assert(std::is_same_v<
                          std::remove_cv_t<std::remove_pointer_t<
                              decltype(std::declval<IndexVector&>().data())>>,
                          Index>,
                      "HybridNeighbourDatabase buffers must contain Index.");

        neighbours.clear();
        if (address == Address{0}) {
            return false;
        }

        const ResolvedAddress resolved = resolve_address(address);
        if (resolved.kind == SegmentKind::Stored) {
            return stored_.read(resolved.physical_address, neighbours);
        }

        using BareVector = std::remove_cv_t<std::remove_reference_t<IndexVector>>;
        if constexpr (std::is_same_v<BareVector, Sigma>) {
            // Normal hot path: let the engine write directly into the caller's
            // recyclable vector, then translate in place without allocation.
            resolved.engine->read_neighbours(
                static_cast<Index>(resolved.local_address),
                neighbours);
            translate_engine_neighbours_in_place(
                resolved.engine->node_count(),
                resolved.node_offset,
                neighbours);
        } else {
            Sigma local_neighbours;
            resolved.engine->read_neighbours(
                static_cast<Index>(resolved.local_address),
                local_neighbours);
            translate_engine_neighbours_in_place(
                resolved.engine->node_count(),
                resolved.node_offset,
                local_neighbours);
            neighbours.assign(local_neighbours.begin(), local_neighbours.end());
        }
        return true;
    }

    [[nodiscard]] std::size_t reserved_unit_count() const noexcept {
        return stored_.reserved_unit_count();
    }

    [[nodiscard]] std::size_t block_count() const {
        return stored_.block_count();
    }

    [[nodiscard]] std::size_t block_unit_length() const noexcept {
        return stored_.block_unit_length();
    }

private:
    template <class EngineT>
    static void validate_node_offset(
        const EngineT& engine,
        Index node_offset) {
        if (engine.node_count() >
            (std::numeric_limits<Index>::max)() - node_offset) {
            throw std::overflow_error(
                "Compute-engine node range exceeds Index capacity.");
        }
    }

    static void translate_engine_neighbours_in_place(
        Index ordinary_count,
        Index node_offset,
        Sigma& neighbours) {
        for (Index& value : neighbours) {
            if (value < ordinary_count) {
                value = static_cast<Index>(node_offset + value);
            }
        }
        // Translation by a fixed offset preserves the order of ordinary indices,
        // but sorting defensively keeps the same canonical stored representation.
        if (neighbours.size() > 1) {
            std::sort(neighbours.begin(), neighbours.end());
        }
    }

    [[nodiscard]] Address append_stored_record(Address physical_address) {
        WriteGuard guard(address_lock_);
        const std::size_t slot = stored_physical_addresses_.size();
        stored_physical_addresses_.push_back(physical_address);

        if (next_address_ == (std::numeric_limits<Address>::max)()) {
            throw std::overflow_error(
                "HybridNeighbourDatabase logical address overflow.");
        }
        const Address logical = next_address_++;

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
            slot,
            Index{0}});
        return logical;
    }

    struct ResolvedAddress {
        SegmentKind kind = SegmentKind::Stored;
        Address physical_address = 0;
        std::shared_ptr<EngineConcept> engine;
        Address local_address = 0;
        Index node_offset = Index{0};
    };

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
                0,
                Index{0}};
        }

        return ResolvedAddress{
            SegmentKind::Engine,
            0,
            engines_.at(segment.source_begin),
            offset,
            segment.node_offset};
    }

    [[nodiscard]] Segment locate_segment_unlocked(Address address) const {
        if (address == Address{0} || segments_.empty()) {
            throw std::out_of_range(
                "Unknown HybridNeighbourDatabase address.");
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
                "Unknown HybridNeighbourDatabase address.");
        }

        const Segment segment = *std::prev(upper);
        if (!segment.contains(address)) {
            throw std::out_of_range(
                "Unknown HybridNeighbourDatabase address.");
        }
        return segment;
    }

    StoreDatabaseType stored_;
    std::vector<std::shared_ptr<EngineConcept>> engines_;
    std::vector<Segment> segments_;
    std::vector<Address> stored_physical_addresses_;
    Address next_address_ = Address{1};
    mutable LockType address_lock_{};
};

} // namespace highvoronoi::detail
