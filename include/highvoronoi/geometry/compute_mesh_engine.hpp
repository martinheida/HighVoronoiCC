#pragma once

/**
 * @file compute_mesh_engine.hpp
 * @brief Common contract for analytic/computed Voronoi node/vertex providers.
 *
 * ComputeMesh is templated on a concrete engine type and owns exactly one engine
 * by value. Consequently, the normal ComputeMesh path does not require runtime
 * polymorphism: adapters inside ComputeMesh refer directly to that concrete
 * engine. This base contract exists so engine types expose a uniform vocabulary
 * and so heterogeneous engines can later be type-erased where runtime
 * polymorphism is actually needed, e.g. inside HybridDataBase or HighVoronoiMesh.
 *
 * Engines own their local node and finite-vertex representation. Local ordinary
 * node indices and local finite-vertex addresses are zero-based. A deleted or
 * inactive finite vertex is represented by an empty sigma on read.
 *
 * Primary/secondary incidence belongs to the engine: every active finite vertex
 * must be primary for exactly one ordinary node and may be secondary for any
 * number of other nodes. Address lists are queried arithmetically and need not
 * be materialized by the engine.
 *
 * If an engine internally uses aliases or multiple local addresses for the same
 * geometric vertex/signature, it is solely responsible for keeping those aliases
 * consistent. In particular, erase_vertex() must not report successful removal
 * while another active alias of the same signature is still meant to exist.
 *
 * Boundary support is optional. The default engine is boundary-agnostic and
 * provides no infinite edges. Implementations may advertise support for a
 * particular Boundary, reconfigure themselves for it, and optionally provide
 * a complete set of infinite edges. This capability is intentionally part of
 * the contract now so later structured engines need no interface break.
 */

#include <highvoronoi/geometry/boundary.hpp>
#include <highvoronoi/geometry/point.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

template <typename NodeScalarT,
          typename VertexScalarT,
          typename IndexT,
          int Dim>
class ComputeMeshEngine {
    static_assert(std::is_integral_v<IndexT> && std::is_unsigned_v<IndexT>,
                  "ComputeMeshEngine Index must be an unsigned integral type.");

public:
    using NodeScalar = NodeScalarT;
    using VertexScalar = VertexScalarT;
    using Index = IndexT;
    using Address = std::size_t;
    using Sigma = std::vector<Index>;
    using NodePoint = std::conditional_t<
        Dim == Dynamic,
        DynamicPoint<NodeScalar>,
        StaticPoint<NodeScalar, Dim>>;
    using VertexPoint = std::conditional_t<
        Dim == Dynamic,
        DynamicPoint<VertexScalar>,
        StaticPoint<VertexScalar, Dim>>;
    using BoundaryType = Boundary<Dim, NodeScalar, Index>;

    static constexpr int DimensionAtCompileTime = Dim;

    virtual ~ComputeMeshEngine() = default;

    ComputeMeshEngine(const ComputeMeshEngine&) = delete;
    ComputeMeshEngine& operator=(const ComputeMeshEngine&) = delete;
    // Concrete engines are commonly moved into ComputeMesh<EngineT>.
    ComputeMeshEngine(ComputeMeshEngine&&) noexcept = default;
    ComputeMeshEngine& operator=(ComputeMeshEngine&&) noexcept = default;

    [[nodiscard]] Index dimension() const noexcept {
        return runtime_dimension_;
    }

    // ------------------------------------------------------------------
    // Ordinary nodes
    // ------------------------------------------------------------------

    [[nodiscard]] virtual Index node_count() const noexcept = 0;

    virtual void copy_node(
        Index local_node,
        NodeScalar* target) const = 0;

    // ------------------------------------------------------------------
    // Finite vertices
    // ------------------------------------------------------------------

    /** Number of locally reserved zero-based finite-vertex slots. */
    [[nodiscard]] virtual Address
    vertex_address_capacity() const noexcept = 0;

    /**
     * Read one local finite vertex. An inactive/deleted slot clears sigma.
     * Active sigma entries use local ordinary-node indices, except for the
     * standard high-end Boundary encodings which are already globally stable.
     */
    virtual void read_vertex(
        Address local_address,
        VertexPoint& position,
        Sigma& sigma) const = 0;

    /**
     * Delete a finite vertex. If an implementation internally aliases one
     * geometric vertex, it is responsible for removing all aliases required to
     * keep the engine's active-signature set unique and consistent.
     */
    [[nodiscard]] virtual bool
    erase_vertex(Address local_address) = 0;

    /** Primary finite-vertex incidence of one ordinary node. */
    [[nodiscard]] virtual std::size_t
    primary_vertex_count(Index local_node) const = 0;

    [[nodiscard]] virtual Address
    primary_vertex_address(
        Index local_node,
        std::size_t ordinal) const = 0;

    /** Secondary finite-vertex incidence of one ordinary node. */
    [[nodiscard]] virtual std::size_t
    secondary_vertex_count(Index local_node) const = 0;

    [[nodiscard]] virtual Address
    secondary_vertex_address(
        Index local_node,
        std::size_t ordinal) const = 0;

    /**
     * Enumerate active finite vertices for HybridDataBase registration.
     * Sparse engines may override this with a faster implementation.
     */
    [[nodiscard]] virtual bool next_active_vertex(
        Address& cursor,
        Address& local_address,
        VertexPoint& position,
        Sigma& sigma) const {
        const Address capacity = vertex_address_capacity();
        while (cursor < capacity) {
            const Address candidate = cursor++;
            sigma.clear();
            read_vertex(candidate, position, sigma);
            if (!sigma.empty()) {
                local_address = candidate;
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Optional boundary / infinite-edge support
    // ------------------------------------------------------------------

    /** True only if this engine knows how to adapt itself to this boundary. */
    [[nodiscard]] virtual bool supports_boundary(
        const BoundaryType&) const noexcept {
        return false;
    }

    /**
     * Reconfigure finite/boundary data for a supported boundary.
     * Implementations that return false from supports_boundary keep the default.
     */
    virtual void configure_boundary(const BoundaryType&) {
        throw std::logic_error(
            "This ComputeMeshEngine does not support boundary computation.");
    }

    /**
     * Whether the engine provides the complete set of infinite Voronoi edges
     * for its current configuration. A caller may use this to distinguish a
     * geometrically consistent finite mesh from a globally complete one.
     */
    [[nodiscard]] virtual bool
    provides_complete_infinite_edges() const noexcept {
        return false;
    }

    [[nodiscard]] virtual Address
    infinite_edge_count() const noexcept {
        return Address{0};
    }

    virtual void read_infinite_edge(
        Address,
        VertexPoint&,
        Sigma& sigma,
        VertexPoint&) const {
        sigma.clear();
        throw std::logic_error(
            "This ComputeMeshEngine does not provide infinite edges.");
    }

    [[nodiscard]] virtual bool erase_infinite_edge(Address) {
        return false;
    }

protected:
    explicit ComputeMeshEngine(Index runtime_dimension)
        : runtime_dimension_(checked_dimension(runtime_dimension)) {}

private:
    [[nodiscard]] static Index checked_dimension(Index dimension) {
        if constexpr (Dim == Dynamic) {
            if (dimension == Index{0}) {
                throw std::invalid_argument(
                    "Dynamic ComputeMeshEngine requires positive dimension.");
            }
            return dimension;
        } else {
            if (dimension != static_cast<Index>(Dim)) {
                throw std::invalid_argument(
                    "ComputeMeshEngine runtime dimension does not match Dim.");
            }
            return static_cast<Index>(Dim);
        }
    }

    Index runtime_dimension_;
};

} // namespace highvoronoi
