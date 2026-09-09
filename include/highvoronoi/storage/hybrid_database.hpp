
#pragma once

/**
 * @file hybrid_database.hpp
 * @brief Thin address-routing extension around one stored database backend.
 *
 * HybridDataBase owns its stored backend by value. Stored records are delegated
 * directly to that backend. Compute engines only contribute logical address
 * ranges; their active signatures are registered in the stored backend's own
 * signature hash through register_signature()/erase_signature(). There is no
 * second hash table in HybridDataBase.
 *
 * The public logical address space is one-based and dense across record slots:
 * stored records, engine-reserved ranges, then later stored records may follow
 * each other arbitrarily. Physical stored-database addresses therefore require
 * one compact logical-slot -> physical-address table.
 */

#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/core/detail/locks.hpp>
#include <highvoronoi/core/point.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

template <
    class Lock,
    class DataBaseParams,
    int Dim,
    class StoreDataBaseT = HVDataBase<Lock, DataBaseParams, Dim>>
class HybridDataBase final {
public:
    using Parameters = DataBaseParams;
    using StoreDataBaseType = StoreDataBaseT;
    using StoredDatabase = StoreDataBaseType;
    using Scalar = typename StoreDataBaseType::Scalar;
    using Index = typename StoreDataBaseType::Index;
    using LockType = Lock;
    using QueueHash = typename StoreDataBaseType::QueueHash;
    using DataUnit = typename StoreDataBaseType::DataUnit;
    using size_t = std::size_t;
    using address_type = std::size_t;
    using Address = address_type;
    using Sigma = typename StoreDataBaseType::Sigma;
    using VertexPoint = typename StoreDataBaseType::VertexPoint;
    using ReadGuard = detail::ReadLockGuard<LockType>;
    using WriteGuard = detail::WriteLockGuard<LockType>;

    static constexpr int DimensionAtCompileTime = Dim;

    static constexpr std::size_t scalar_unit_count =
        StoreDataBaseType::scalar_unit_count;
    static constexpr std::size_t index_unit_count =
        StoreDataBaseType::index_unit_count;

    static_assert(std::is_same_v<typename StoreDataBaseType::Scalar, Scalar>);
    static_assert(std::is_same_v<typename StoreDataBaseType::Index, Index>);
    static_assert(StoreDataBaseType::DimensionAtCompileTime == Dim);
    static_assert(std::is_same_v<typename StoreDataBaseType::Sigma, Sigma>);
    static_assert(std::is_same_v<typename StoreDataBaseType::VertexPoint, VertexPoint>);

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
        [[nodiscard]] virtual Address capacity() const noexcept = 0;
        [[nodiscard]] virtual Index node_count() const noexcept = 0;
        virtual void read_vertex(
            Address local_address,
            VertexPoint& position,
            Sigma& sigma) const = 0;
        [[nodiscard]] virtual bool erase_vertex(Address local_address) = 0;
    };

    template <class EngineT>
    class EngineModel final : public EngineConcept {
    public:
        using Engine = EngineT;
        using VertexPoint = typename Engine::VertexPoint;

        explicit EngineModel(std::shared_ptr<Engine> engine)
            : engine_(std::move(engine)) {}

        [[nodiscard]] Address capacity() const noexcept override {
            return static_cast<Address>(engine_->vertex_address_capacity());
        }

        [[nodiscard]] Index node_count() const noexcept override {
            return engine_->node_count();
        }

        void read_vertex(
            Address local_address,
            VertexPoint& position,
            Sigma& sigma) const override {
            engine_->read_vertex(local_address, position, sigma);
        }

        [[nodiscard]] bool erase_vertex(Address local_address) override {
            return engine_->erase_vertex(local_address);
        }

    private:
        std::shared_ptr<Engine> engine_;
    };

    struct Segment {
        Address begin = 0; // inclusive
        Address end = 0;   // exclusive
        SegmentKind kind = SegmentKind::Stored;
        std::size_t source_begin = 0; // stored-address slot or engine index
        Index node_offset = Index{0};

        [[nodiscard]] bool contains(Address address) const noexcept {
            return address >= begin && address < end;
        }
    };

public:
    explicit HybridDataBase(
        std::size_t unit_length,
        const DataBaseParams& parameters,
        std::size_t runtime_dimension =
            Dim == Dynamic ? std::size_t{0} : static_cast<std::size_t>(Dim))
        : stored_(unit_length, parameters, runtime_dimension) {}

    HybridDataBase(const HybridDataBase&) = delete;
    HybridDataBase& operator=(const HybridDataBase&) = delete;
    HybridDataBase(HybridDataBase&&) = delete;
    HybridDataBase& operator=(HybridDataBase&&) = delete;

    [[nodiscard]] StoreDataBaseType& stored_database() noexcept {
        return stored_;
    }

    [[nodiscard]] const StoreDataBaseType& stored_database() const noexcept {
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

    template <class EngineT>
    [[nodiscard]] EngineRegistration register_engine(
        std::shared_ptr<EngineT> engine,
        Index node_offset) {
        static_assert(std::is_same_v<typename EngineT::VertexScalar, Scalar>,
                      "Engine VertexScalar must match database Scalar.");
        static_assert(std::is_same_v<typename EngineT::Index, Index>,
                      "Engine Index must match database Index.");
        static_assert(EngineT::DimensionAtCompileTime == Dim,
                      "Engine dimension must match database dimension.");
        static_assert(std::is_same_v<typename EngineT::VertexPoint, VertexPoint>,
                      "Engine VertexPoint must match database VertexPoint.");
        static_assert(std::is_same_v<typename EngineT::Sigma, Sigma>,
                      "Engine Sigma must match database Sigma.");

        if (!engine) {
            throw std::invalid_argument(
                "HybridDataBase requires a non-null engine.");
        }
        validate_node_offset(*engine, node_offset);

        const Address capacity =
            static_cast<Address>(engine->vertex_address_capacity());
        std::shared_ptr<EngineConcept> model =
            std::make_shared<EngineModel<EngineT>>(engine);

        {
            WriteGuard guard(address_lock_);
            if (capacity >
                (std::numeric_limits<Address>::max)() - next_address_) {
                throw std::overflow_error(
                    "HybridDataBase logical address overflow.");
            }
            engines_.reserve(engines_.size() + 1);
            if (capacity != 0) {
                segments_.reserve(segments_.size() + 1);
            }
        }

        std::vector<Sigma> inserted_signatures;
        VertexPoint position = make_vertex_point();
        Sigma local_sigma;
        Sigma global_sigma;
        Address cursor = 0;
        Address local_address = 0;

        try {
            while (engine->next_active_vertex(
                       cursor,
                       local_address,
                       position,
                       local_sigma)) {
                if (local_address >= capacity) {
                    throw std::logic_error(
                        "Compute engine enumerated a vertex outside its reserved range.");
                }
                translate_engine_signature(
                    *engine,
                    node_offset,
                    local_sigma,
                    global_sigma);
                if (global_sigma.empty()) {
                    throw std::logic_error(
                        "Active compute-engine vertex returned an empty sigma.");
                }
                if (!stored_.register_signature(global_sigma)) {
                    throw std::logic_error(
                        "Compute-engine registration encountered an existing global vertex signature.");
                }
                inserted_signatures.push_back(global_sigma);
            }
        } catch (...) {
            for (const Sigma& sigma : inserted_signatures) {
                (void)stored_.erase_signature(sigma);
            }
            throw;
        }

        WriteGuard guard(address_lock_);
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

    [[nodiscard]] Address push(
        const VertexPoint& position,
        const Sigma& sigma) {
        const Address physical = stored_.push(position, sigma);
        if (physical == Address{0}) {
            return Address{0};
        }
        return append_stored_record(physical);
    }

    void read(
        Address address,
        VertexPoint& position,
        Sigma& sigma) const {
        const ResolvedAddress resolved = resolve_address(address);
        if (resolved.kind == SegmentKind::Stored) {
            stored_.read(resolved.physical_address, position, sigma);
            return;
        }

        if constexpr (Dim == Dynamic) {
            position.resize(static_cast<Eigen::Index>(stored_.dimension()));
        }
        resolved.engine->read_vertex(
            resolved.local_address,
            position,
            sigma);
        if (sigma.empty()) {
            return;
        }
        translate_engine_signature_in_place(
            resolved.engine->node_count(),
            resolved.node_offset,
            sigma);
    }

    [[nodiscard]] bool contains(const Sigma& sigma) const {
        return stored_.contains(sigma);
    }

    [[nodiscard]] bool register_signature(const Sigma& sigma) {
        return stored_.register_signature(sigma);
    }

    [[nodiscard]] bool erase_signature(const Sigma& sigma) {
        return stored_.erase_signature(sigma);
    }

    bool erase(Address address, const Sigma& sigma) {
        const ResolvedAddress resolved = resolve_address(address);
        if (resolved.kind == SegmentKind::Stored) {
            return stored_.erase(resolved.physical_address, sigma);
        }

        if (!resolved.engine->erase_vertex(resolved.local_address)) {
            return false;
        }
        if (!stored_.erase_signature(sigma)) {
            throw std::logic_error(
                "HybridDataBase erased engine vertex missing from stored hash.");
        }
        return true;
    }

    [[nodiscard]] Address push_facet(
        const VertexPoint& origin,
        Sigma& sigma,
        const VertexPoint& direction) {
        const Address physical = stored_.push_facet(origin, sigma, direction);
        if (physical == Address{0}) {
            return Address{0};
        }
        return append_stored_record(physical);
    }

    void read_facet(
        Address address,
        VertexPoint& origin,
        Sigma& sigma,
        VertexPoint& direction) const {
        const ResolvedAddress resolved = resolve_address(address);
        if (resolved.kind != SegmentKind::Stored) {
            throw std::invalid_argument(
                "Compute-engine address cannot be read as a facet record.");
        }
        stored_.read_facet(
            resolved.physical_address,
            origin,
            sigma,
            direction);
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
    [[nodiscard]] VertexPoint make_vertex_point() const {
        if constexpr (Dim == Dynamic) {
            return VertexPoint(static_cast<Eigen::Index>(stored_.dimension()));
        } else {
            return VertexPoint{};
        }
    }

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

    template <class EngineT>
    static void translate_engine_signature(
        const EngineT& engine,
        Index node_offset,
        const Sigma& local_sigma,
        Sigma& global_sigma) {
        translate_engine_signature(
            engine.node_count(),
            node_offset,
            local_sigma,
            global_sigma);
    }

    static void translate_engine_signature(
        Index ordinary_count,
        Index node_offset,
        const Sigma& local_sigma,
        Sigma& global_sigma) {
        global_sigma.clear();
        global_sigma.reserve(local_sigma.size());
        for (const Index value : local_sigma) {
            if (value < ordinary_count) {
                global_sigma.push_back(
                    static_cast<Index>(node_offset + value));
            } else {
                // High-end Boundary mirror encodings are already global.
                global_sigma.push_back(value);
            }
        }
        if (global_sigma.size() > 1) {
            std::sort(global_sigma.begin(), global_sigma.end());
        }
    }

    static void translate_engine_signature_in_place(
        Index ordinary_count,
        Index node_offset,
        Sigma& sigma) {
        for (Index& value : sigma) {
            if (value < ordinary_count) {
                value = static_cast<Index>(node_offset + value);
            }
        }
        if (sigma.size() > 1) {
            std::sort(sigma.begin(), sigma.end());
        }
    }

    [[nodiscard]] Address append_stored_record(Address physical_address) {
        WriteGuard guard(address_lock_);
        const std::size_t slot = stored_physical_addresses_.size();
        stored_physical_addresses_.push_back(physical_address);

        if (next_address_ == (std::numeric_limits<Address>::max)()) {
            throw std::overflow_error(
                "HybridDataBase logical address overflow.");
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
            throw std::out_of_range("Unknown HybridDataBase address.");
        }
        const auto upper = std::upper_bound(
            segments_.begin(),
            segments_.end(),
            address,
            [](Address value, const Segment& segment) {
                return value < segment.begin;
            });
        if (upper == segments_.begin()) {
            throw std::out_of_range("Unknown HybridDataBase address.");
        }
        const Segment segment = *std::prev(upper);
        if (!segment.contains(address)) {
            throw std::out_of_range("Unknown HybridDataBase address.");
        }
        return segment;
    }

    StoreDataBaseType stored_;
    std::vector<std::shared_ptr<EngineConcept>> engines_;
    std::vector<Segment> segments_;
    std::vector<Address> stored_physical_addresses_;
    Address next_address_ = Address{1};
    mutable LockType address_lock_{};
};

} // namespace highvoronoi


