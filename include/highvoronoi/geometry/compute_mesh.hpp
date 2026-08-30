#pragma once

/**
 * @file compute_mesh.hpp
 * @brief Read-oriented AbstractMesh facade owning one concrete compute engine.
 *
 * ComputeMesh intentionally does not materialize computed vertices. Its database
 * adapter exposes the existing AbstractMesh read/erase/contains contract while
 * finite vertex addresses are one-based facade addresses corresponding to the
 * engine's zero-based local vertex slots. Primary and secondary incidence lists
 * are lightweight arithmetic adapters over the engine.
 *
 * This class is intended as an import/source mesh and validation facade. It does
 * not support storing new finite vertices or infinite edges into the engine.
 */

#include <highvoronoi/detail/locks.hpp>
#include <highvoronoi/geometry/abstract_mesh.hpp>
#include <highvoronoi/geometry/compute_mesh_engine.hpp>
#include <highvoronoi/geometry/mesh_index_mapping.hpp>
#include <highvoronoi/geometry/voronoi_nodes.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace highvoronoi {

namespace detail {

/**
 * @brief Selects which engine-owned incidence list is exposed as an AbstractMesh address list.
 *
 * Finite engine vertex slots are zero-based inside the engine but are presented as one-based
 * mesh/database addresses by EngineAddressList. Infinite-edge addresses, when provided by the
 * engine, follow the reserved finite-vertex address range.
 */
enum class EngineAddressListKind : unsigned char {
    Primary,
    Secondary,
    Infinite
};

/**
 * @brief Lightweight, non-owning adapter exposing engine incidences as an address-list object.
 *
 * The adapter stores references to exactly one concrete engine. It does not allocate or copy
 * vertex addresses: size() and operator[] query the engine directly. Primary and secondary
 * incidence are therefore determined entirely by the engine implementation.
 */
template <class EngineT>
class EngineAddressList {
public:
    using Engine = EngineT;
    using Index = typename Engine::Index;
    using Address = typename Engine::Address;

    EngineAddressList(
        Engine& engine,
        EngineAddressListKind kind,
        Index node = Index{0})
        : engine_(engine), kind_(kind), node_(node) {}

    [[nodiscard]] std::size_t size() const noexcept {
                switch (kind_) {
        case EngineAddressListKind::Primary:
            return engine_.primary_vertex_count(node_);
        case EngineAddressListKind::Secondary:
            return engine_.secondary_vertex_count(node_);
        case EngineAddressListKind::Infinite:
            return engine_.provides_complete_infinite_edges()
                ? static_cast<std::size_t>(engine_.infinite_edge_count())
                : 0u;
        }
        return 0;
    }

    [[nodiscard]] std::size_t operator[](std::size_t position) const {
                const std::size_t count = size();
        if (position >= count) {
            throw std::out_of_range("Engine address-list position out of range.");
        }

        switch (kind_) {
        case EngineAddressListKind::Primary:
            return one_based(engine_.primary_vertex_address(node_, position));
        case EngineAddressListKind::Secondary:
            return one_based(engine_.secondary_vertex_address(node_, position));
        case EngineAddressListKind::Infinite:
            return one_based_infinite(static_cast<Address>(position));
        }
        throw std::logic_error("Unknown engine address-list kind.");
    }

private:
    [[nodiscard]] static std::size_t one_based(Address local) {
        return static_cast<std::size_t>(local) + 1u;
    }

    [[nodiscard]] std::size_t one_based_infinite(Address local) const {
        return static_cast<std::size_t>(engine_.vertex_address_capacity()) +
               static_cast<std::size_t>(local) + 1u;
    }

    Engine& engine_;
    EngineAddressListKind kind_ = EngineAddressListKind::Primary;
    Index node_ = Index{0};
};

/**
 * @brief Read/erase database facade over one concrete ComputeMesh engine.
 *
 * This adapter exists so the existing AbstractMesh vertex iterators can consume computed
 * vertices through the familiar database interface. No finite vertex is materialized here.
 * Engine-local finite vertex addresses are zero-based; the facade exposes them as one-based
 * addresses. An inactive/deleted engine vertex is represented by an empty sigma, exactly like
 * a tombstoned stored database record.
 *
 * push() and push_facet() intentionally throw: ComputeMesh is a source/read facade, not a
 * mutable storage mesh. contains() enumerates active engine vertices and erase() routes the
 * request back to the engine.
 */
template <class EngineT>
class ComputeMeshDatabaseAdapter {
public:
    using Engine = EngineT;
    using Scalar = typename Engine::VertexScalar;
    using Index = typename Engine::Index;
    using Address = std::size_t;
    using LockType = detail::EmptyLock;
    using Sigma = typename Engine::Sigma;
    using VertexPoint = typename Engine::VertexPoint;

    explicit ComputeMeshDatabaseAdapter(Engine& engine) noexcept
        : engine_(engine) {}

    template <class RVector, class SigmaVector>
    [[nodiscard]] Address push(const RVector&, const SigmaVector&) {
        throw std::logic_error(
            "ComputeMesh is read-oriented; computed vertices cannot be pushed into its engine facade.");
    }

    template <class RVector, class SigmaVector>
    void read(Address address, RVector& position, SigmaVector& sigma) const {
        const auto local = finite_local_address(address);
        VertexPoint local_position = make_vertex_point();
        Sigma local_sigma;
        engine_.read_vertex(local, local_position, local_sigma);
        sigma.clear();
        if (local_sigma.empty()) {
            return;
        }
        copy_point(local_position, position);
        sigma.resize(local_sigma.size());
        std::copy(local_sigma.begin(), local_sigma.end(), sigma.begin());
    }

    template <class SigmaVector>
    [[nodiscard]] bool contains(const SigmaVector& sigma) const {
        VertexPoint position = make_vertex_point();
        Sigma candidate;
        Address cursor = 0;
        Address local = 0;
        while (engine_.next_active_vertex(cursor, local, position, candidate)) {
            if (candidate.size() == sigma.size() &&
                std::equal(candidate.begin(), candidate.end(), sigma.begin())) {
                return true;
            }
        }
        return false;
    }

    template <class SigmaVector>
    [[nodiscard]] bool erase(Address address, const SigmaVector&) {
        const Address finite_capacity = engine_.vertex_address_capacity();
        if (address >= 1 && address <= finite_capacity) {
            return engine_.erase_vertex(address - 1);
        }
        const Address first_infinite = finite_capacity + 1;
        const Address infinite_count = engine_.infinite_edge_count();
        if (address >= first_infinite &&
            address < first_infinite + infinite_count) {
            return engine_.erase_infinite_edge(address - first_infinite);
        }
        return false;
    }

    template <class RVector, class SigmaVector, class UVector>
    [[nodiscard]] Address push_facet(
        const RVector&,
        SigmaVector&,
        const UVector&) {
        throw std::logic_error(
            "ComputeMesh cannot push infinite edges into its engine facade.");
    }

    template <class RVector, class SigmaVector, class UVector>
    void read_facet(
        Address address,
        RVector& origin,
        SigmaVector& sigma,
        UVector& direction) const {
        const Address finite_capacity = engine_.vertex_address_capacity();
        const Address first_infinite = finite_capacity + 1;
        const Address infinite_count = engine_.infinite_edge_count();
        if (address < first_infinite ||
            address >= first_infinite + infinite_count) {
            throw std::out_of_range("ComputeMesh infinite-edge address out of range.");
        }

        VertexPoint local_origin = make_vertex_point();
        VertexPoint local_direction = make_vertex_point();
        Sigma local_sigma;
        engine_.read_infinite_edge(
            address - first_infinite,
            local_origin,
            local_sigma,
            local_direction);
        sigma.clear();
        if (local_sigma.empty()) {
            return;
        }
        copy_point(local_origin, origin);
        copy_point(local_direction, direction);
        sigma.resize(local_sigma.size());
        std::copy(local_sigma.begin(), local_sigma.end(), sigma.begin());
    }

private:
    [[nodiscard]] Address finite_local_address(Address address) const {
        if (address == 0 || address > engine_.vertex_address_capacity()) {
            throw std::out_of_range("ComputeMesh finite-vertex address out of range.");
        }
        return address - 1;
    }

    [[nodiscard]] VertexPoint make_vertex_point() const {
        if constexpr (Engine::DimensionAtCompileTime == Dynamic) {
            return VertexPoint(static_cast<Eigen::Index>(engine_.dimension()));
        } else {
            return VertexPoint{};
        }
    }

    template <class Source, class Target>
    static void copy_point(const Source& source, Target& target) {
        std::copy(
            source.data(),
            source.data() + source.size(),
            target.data());
    }

    Engine& engine_;
};

/**
 * @brief Computed-node access facade backed by a concrete engine and the mesh index mapping.
 *
 * Public node indices are translated through DenseIndexMapping before the engine is asked to
 * compute/copy a node. The class stores references only; the owning ComputeMesh owns the engine
 * and the mapping for the complete lifetime of this facade.
 */
template <class EngineT>
class PublicComputedEngineNodes final
    : public ComputedNodeAccess<
          typename EngineT::NodeScalar,
          EngineT::DimensionAtCompileTime,
          typename EngineT::Index> {
public:
    using Engine = EngineT;
    using Scalar = typename Engine::NodeScalar;
    using Index = typename Engine::Index;
    static constexpr int Dim = Engine::DimensionAtCompileTime;
    using Base = ComputedNodeAccess<Scalar, Dim, Index>;
    using Mapping = DenseIndexMapping<Index>;

    PublicComputedEngineNodes(
        Engine& engine,
        const Mapping& mapping)
        : Base(engine.dimension()),
          engine_(engine),
          mapping_(mapping) {}

    [[nodiscard]] Index size() const noexcept override {
        return mapping_.size();
    }

    [[nodiscard]] Scalar get_data(
        Index public_node,
        Index coordinate) const override {
        if (coordinate >= this->dimension()) {
            throw std::out_of_range(
                "Computed engine node coordinate out of range.");
        }
        typename Base::Point point = this->make_point();
        const Index internal = mapping_.public_to_internal(public_node);
        engine_.copy_node(internal, point.data());
        return point[coordinate];
    }

private:
    void compute_node(Index public_node, Scalar* target) const override {
        const Index internal = mapping_.public_to_internal(public_node);
        engine_.copy_node(internal, target);
    }

    Engine& engine_;
    const Mapping& mapping_;
};

} // namespace detail

/**
 * @brief AbstractMesh facade owning exactly one concrete compute engine.
 *
 * EngineT is a concrete compile-time engine type, not a polymorphic base. ComputeMesh takes the
 * engine by rvalue reference and owns it by value. All internal helper objects keep references
 * to that owned engine, so there is no shared ownership and no nullable engine state.
 *
 * The mesh exposes the engine's computed nodes and vertices through the normal AbstractMesh API:
 * primary_vertices(), secondary_vertices(), vertices(), erase_vertex(), boundary(), and the
 * search-tree node access all operate without materializing finite vertices in an HVDataBase.
 *
 * Primary/secondary classification is delegated to EngineT through primary_vertex_* and
 * secondary_vertex_* queries. Boundary support is optional: a non-empty Boundary can only be
 * installed when EngineT::supports_boundary() accepts it. Likewise, infinite-edge iteration is
 * only complete when EngineT::provides_complete_infinite_edges() reports true.
 *
 * ComputeMesh is intended as a computed source/import mesh. New Voronoi vertices created by
 * later interaction with unrelated nodes belong in a stored/hybrid mesh, not in this facade.
 */
template <class EngineT>
class ComputeMesh final
    : public AbstractMesh<
          typename EngineT::NodeScalar,
          typename EngineT::VertexScalar,
          typename EngineT::Index,
          EngineT::DimensionAtCompileTime,
          NodeAccessMode::Computed,
          detail::ComputeMeshDatabaseAdapter<EngineT>,
          detail::EngineAddressList<EngineT>,
          DenseIndexMapping<typename EngineT::Index>> {
public:
    using Engine = EngineT;
    using NodeScalar = typename Engine::NodeScalar;
    using VertexScalar = typename Engine::VertexScalar;
    using Index = typename Engine::Index;
    using Database = detail::ComputeMeshDatabaseAdapter<Engine>;
    using AddressList = detail::EngineAddressList<Engine>;
    using IndexMapping = DenseIndexMapping<Index>;
    static constexpr int Dim = Engine::DimensionAtCompileTime;

    using Base = AbstractMesh<
        NodeScalar,
        VertexScalar,
        Index,
        Dim,
        NodeAccessMode::Computed,
        Database,
        AddressList,
        IndexMapping>;

    using BoundaryType = typename Base::BoundaryType;
    using NodesAccess = typename Base::NodesAccess;
    using ExtendedNodesAccess = typename Base::ExtendedNodesAccess;
    using PublicNodes = detail::PublicComputedEngineNodes<Engine>;
    using ExtendedNodes = ExtendedVoronoiNodes<PublicNodes>;
    using Address = typename Base::Address;

    /**
     * @brief Move one concrete engine into this mesh.
     *
     * The engine becomes exclusively owned by the ComputeMesh. Internal node, address-list,
     * and database adapters reference this member directly. The optional boundary is forwarded
     * to the engine only when it is non-empty and supported.
     */
    explicit ComputeMesh(
        Engine&& engine,
        BoundaryType boundary = BoundaryType{})
        : Base(
              engine.dimension(),
              IndexMapping::identity(engine.node_count())),
          engine_(std::move(engine)),
          database_(engine_),
          public_nodes_(engine_, this->index_mapping()),
          extended_nodes_(
              std::in_place,
              public_nodes_,
              prepare_boundary(std::move(boundary))),
          infinite_addresses_(
              engine_,
              detail::EngineAddressListKind::Infinite) {
        build_node_address_lists();
    }

    /** @brief Access the owned concrete engine. */
    [[nodiscard]] Engine& engine() noexcept {
        return engine_;
    }

    [[nodiscard]] const Engine& engine() const noexcept {
        return engine_;
    }

    /** @brief Query whether the owned engine can represent the supplied boundary. */
    [[nodiscard]] bool supports_boundary(
        const BoundaryType& boundary) const noexcept {
        return engine_.supports_boundary(boundary);
    }

    /**
     * @brief Report whether engine-provided infinite-edge iteration is complete.
     *
     * A false result does not invalidate finite vertices; it only means full mesh completeness
     * checks requiring all unbounded rays cannot be satisfied by this ComputeMesh alone.
     */
    [[nodiscard]] bool
    provides_complete_infinite_edges() const noexcept {
        return engine_.provides_complete_infinite_edges();
    }

    /** Concrete extended-node access for SearchTree safe-copy/rebind traits. */
    [[nodiscard]] ExtendedNodes& concrete_extended_nodes() noexcept {
        return *extended_nodes_;
    }

    [[nodiscard]] const ExtendedNodes&
    concrete_extended_nodes() const noexcept {
        return *extended_nodes_;
    }

private:

    [[nodiscard]] BoundaryType prepare_boundary(BoundaryType boundary) {
        if (boundary.size() != Index{0}) {
            if (!engine_.supports_boundary(boundary)) {
                throw std::invalid_argument(
                    "ComputeMeshEngine does not support the requested non-empty boundary.");
            }
            engine_.configure_boundary(boundary);
        }
        return boundary;
    }

    void build_node_address_lists() {
        const auto count = static_cast<std::size_t>(engine_.node_count());
        primary_.reserve(count);
        secondary_.reserve(count);
        for (Index node = 0; node < engine_.node_count(); ++node) {
            primary_.emplace_back(
                engine_, detail::EngineAddressListKind::Primary, node);
            secondary_.emplace_back(
                engine_, detail::EngineAddressListKind::Secondary, node);
        }
    }

    [[nodiscard]] const NodesAccess& nodes_impl() const noexcept override {
        return extended_nodes_->inner_nodes();
    }

    [[nodiscard]] ExtendedNodesAccess& extended_nodes_impl() noexcept override {
        return *extended_nodes_;
    }

    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept override {
        return *extended_nodes_;
    }

    [[nodiscard]] const BoundaryType& boundary_impl() const noexcept override {
        return extended_nodes_->boundary();
    }

    void set_boundary_impl(BoundaryType boundary) override {
        if (boundary.size() != Index{0}) {
            if (!engine_.supports_boundary(boundary)) {
                throw std::invalid_argument(
                    "ComputeMeshEngine does not support the requested boundary.");
            }
            engine_.configure_boundary(boundary);
        }
        extended_nodes_.emplace(public_nodes_, std::move(boundary));
    }

    [[nodiscard]] Database& database_impl() noexcept override {
        return database_;
    }

    [[nodiscard]] const Database& database_impl() const noexcept override {
        return database_;
    }

    [[nodiscard]] Index internal_node_count_impl() const noexcept override {
        return engine_.node_count();
    }

    [[nodiscard]] Index
    public_node_to_internal_impl(Index public_node) const override {
        return this->index_mapping().public_to_internal(public_node);
    }

    [[nodiscard]] std::optional<Index>
    internal_node_to_public_impl(Index internal_node) const override {
        return this->index_mapping().internal_to_public(internal_node);
    }

    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        return primary_.at(static_cast<std::size_t>(internal_node));
    }

    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return secondary_.at(static_cast<std::size_t>(internal_node));
    }

    void register_primary_vertex_impl(Index, Address) override {
        throw std::logic_error(
            "ComputeMesh cannot register stored vertices in its computed engine.");
    }

    void register_secondary_vertex_impl(Index, Address) override {
        throw std::logic_error(
            "ComputeMesh cannot register stored vertices in its computed engine.");
    }

    [[nodiscard]] const AddressList&
    infinite_edge_addresses_impl() const override {
        return infinite_addresses_;
    }

    void register_infinite_edge_impl(Address) override {
        throw std::logic_error(
            "ComputeMesh cannot register stored infinite edges in its computed engine.");
    }

    /**
     * @brief Remove a node from the public mesh numbering without changing engine-local identity.
     *
     * DenseIndexMapping keeps the stable internal index while cleanup_vertex_lists_impl()
     * rebuilds the public numbering after deletions.
     */
    void mark_internal_node_deleted_impl(Index internal_node) override {
        this->index_mapping().mark_deleted(internal_node);
    }

    void cleanup_vertex_lists_impl() override {
        this->index_mapping().rebuild();
    }

    Engine engine_;
    Database database_;
    PublicNodes public_nodes_;
    std::optional<ExtendedNodes> extended_nodes_;
    std::vector<AddressList> primary_;
    std::vector<AddressList> secondary_;
    AddressList infinite_addresses_;
};

} // namespace highvoronoi
