#pragma once

/**
 * @file integral_engine.hpp
 * @brief One coherent area/integral engine layered on top of one mesh engine.
 *
 * A user implements exactly one IntegralEngine subclass.  Area and integral
 * databases see that same engine through one internal adapter object; callers
 * do not need to implement separate area/integral engines.
 */

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @brief Base class for computed measure/integral data attached to a mesh engine.
 *
 * The two record spaces remain semantically separate:
 *   - area records: one scalar per neighbour occurrence;
 *   - integral records: bulk or neighbour-major interface values.
 *
 * A concrete engine may compute either part analytically, but it is one coherent
 * object and retains/references exactly one MeshEngine.
 */
template <class MeshEngineT,
          typename AreaScalarT,
          typename IntegralScalarT>
class IntegralEngine {
public:
    using MeshEngine = MeshEngineT;
    using AreaScalar = AreaScalarT;
    using IntegralScalar = IntegralScalarT;
    using Index = typename MeshEngine::Index;
    using Address = std::size_t;
    using AreaBuffer = std::vector<AreaScalar>;
    using IntegralBuffer = std::vector<IntegralScalar>;

    static_assert(
        std::is_integral_v<Index> && std::is_unsigned_v<Index>,
        "IntegralEngine mesh Index must be an unsigned integral type");
    static_assert(
        std::is_floating_point_v<AreaScalar>,
        "IntegralEngine AreaScalar must be floating-point");
    static_assert(
        std::is_floating_point_v<IntegralScalar>,
        "IntegralEngine IntegralScalar must be floating-point");

    virtual ~IntegralEngine() = default;

    /** Non-owning attachment. MeshEngine must outlive this IntegralEngine. */
    explicit IntegralEngine(MeshEngine& mesh_engine) noexcept
        : mesh_engine_(&mesh_engine) {}

    /** Shared-owning attachment. */
    explicit IntegralEngine(std::shared_ptr<MeshEngine> mesh_engine)
        : owned_mesh_engine_(std::move(mesh_engine)),
          mesh_engine_(owned_mesh_engine_.get()) {
        if (!mesh_engine_) {
            throw std::invalid_argument(
                "IntegralEngine requires a non-null mesh engine");
        }
    }

    [[nodiscard]] const MeshEngine& mesh_engine() const noexcept {
        return *mesh_engine_;
    }

    [[nodiscard]] MeshEngine& mesh_engine() noexcept {
        return *mesh_engine_;
    }

    [[nodiscard]] Index node_count() const noexcept {
        return mesh_engine_->node_count();
    }

    // The concrete user engine implements both sections below on the same mesh engine.
    // Database-specific splitting happens only inside IntegralEngineDatabaseAdapter.
    // ---------------------------------------------------------------------
    // Cell measure / interface area interface.
    // ---------------------------------------------------------------------

    /** Read the d-dimensional measure of one cell. */
    [[nodiscard]] virtual bool read_volume(
        Index local_node,
        AreaScalar& volume) const = 0;

    /** Number of immutable/reproducible local area-record addresses. */
    [[nodiscard]] virtual Address
    area_address_capacity() const noexcept = 0;

    /** Read one area record, ordinally aligned with the mesh neighbour record. */
    [[nodiscard]] virtual bool read_area(
        Address local_address,
        AreaBuffer& areas) const = 0;

    /** Local area-record address for one cell, if the engine provides it. */
    [[nodiscard]] virtual std::optional<Address>
    area_address(Index local_node) const = 0;

    // ---------------------------------------------------------------------
    // Bulk/interface integral interface.
    // ---------------------------------------------------------------------

    /** Number of components in one user integral value. */
    [[nodiscard]] virtual std::size_t
    integral_components() const noexcept = 0;

    /** Number of immutable/reproducible local integral-record addresses. */
    [[nodiscard]] virtual Address
    integral_address_capacity() const noexcept = 0;

    /** Read one bulk or interface integral record. */
    [[nodiscard]] virtual bool read_integral(
        Address local_address,
        IntegralBuffer& values) const = 0;

    /** Local record containing the cell/bulk integral. */
    [[nodiscard]] virtual std::optional<Address>
    bulk_integral_address(Index local_node) const = 0;

    /**
     * Local record containing all interface integrals of one cell.
     * Layout is neighbour-major and flattened:
     *   [n0 component 0..p-1, n1 component 0..p-1, ...].
     */
    [[nodiscard]] virtual std::optional<Address>
    interface_integral_address(Index local_node) const = 0;

private:
    std::shared_ptr<MeshEngine> owned_mesh_engine_;
    MeshEngine* mesh_engine_ = nullptr;
};

namespace detail {

/** Database-facing area type-erasure contract. */
template <typename AreaScalarT, typename IndexT>
class AreaEngineInterface {
public:
    using AreaScalar = AreaScalarT;
    using Index = IndexT;
    using Address = std::size_t;
    using AreaBuffer = std::vector<AreaScalar>;

    virtual ~AreaEngineInterface() = default;
    [[nodiscard]] virtual Address area_address_capacity() const noexcept = 0;
    [[nodiscard]] virtual bool read_area(
        Address local_address,
        AreaBuffer& areas) const = 0;
};

/** Database-facing integral type-erasure contract. */
template <typename IntegralScalarT, typename IndexT>
class IntegralEngineInterface {
public:
    using IntegralScalar = IntegralScalarT;
    using Index = IndexT;
    using Address = std::size_t;
    using IntegralBuffer = std::vector<IntegralScalar>;

    virtual ~IntegralEngineInterface() = default;
    [[nodiscard]] virtual Address integral_address_capacity() const noexcept = 0;
    [[nodiscard]] virtual bool read_integral(
        Address local_address,
        IntegralBuffer& values) const = 0;
};

/**
 * @brief The single adapter object used to connect one coherent IntegralEngine
 *        to both hybrid numeric databases.
 */
template <class EngineT>
class IntegralEngineDatabaseAdapter final
    : public AreaEngineInterface<
          typename EngineT::AreaScalar,
          typename EngineT::Index>,
      public IntegralEngineInterface<
          typename EngineT::IntegralScalar,
          typename EngineT::Index> {
public:
    using Engine = EngineT;
    using AreaScalar = typename Engine::AreaScalar;
    using IntegralScalar = typename Engine::IntegralScalar;
    using Index = typename Engine::Index;
    using Address = std::size_t;
    using AreaBuffer = std::vector<AreaScalar>;
    using IntegralBuffer = std::vector<IntegralScalar>;

    // This is the only adapter needed by user code: one coherent engine object is
    // presented as the two narrow read interfaces required by the hybrid databases.
    explicit IntegralEngineDatabaseAdapter(
        std::shared_ptr<const Engine> engine)
        : engine_(std::move(engine)) {
        if (!engine_) {
            throw std::invalid_argument(
                "IntegralEngineDatabaseAdapter requires a non-null engine");
        }
    }

    [[nodiscard]] Address area_address_capacity() const noexcept override {
        return engine_->area_address_capacity();
    }

    [[nodiscard]] bool read_area(
        Address local_address,
        AreaBuffer& areas) const override {
        return engine_->read_area(local_address, areas);
    }

    [[nodiscard]] Address integral_address_capacity() const noexcept override {
        return engine_->integral_address_capacity();
    }

    [[nodiscard]] bool read_integral(
        Address local_address,
        IntegralBuffer& values) const override {
        return engine_->read_integral(local_address, values);
    }

    [[nodiscard]] const Engine& engine() const noexcept {
        return *engine_;
    }

private:
    std::shared_ptr<const Engine> engine_;
};

} // namespace detail
} // namespace highvoronoi
