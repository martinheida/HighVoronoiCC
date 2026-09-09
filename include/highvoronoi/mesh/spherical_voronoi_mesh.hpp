
#pragma once

/**
 * @file spherical_voronoi_mesh.hpp
 * @brief Spherical Voronoi mesh obtained from the Euclidean origin cell.
 *
 * The construction is deliberately reduced to an ordinary Euclidean Voronoi
 * problem in R^d. Internal node 0 is the origin. User generators live on the
 * unit sphere and only the Voronoi cell of node 0 is computed. Every stored
 * Euclidean vertex therefore contains internal node 0 in its signature. The
 * public representation removes the origin from the signature and projects the
 * Euclidean vertex radially onto the unit sphere.
 *
 * Neighbour classification likewise reuses the ordinary Euclidean neighbour
 * kernel on each non-origin internal cell.  The stored internal record keeps
 * origin node 0 as a genuine Euclidean neighbour.  Only the spherical public
 * projection removes node 0.  Antipodal public projection maps q and -q to the
 * same visible neighbour while preserving multiplicity of distinct interfaces.
 *
 * In AntipodalHemisphere mode every visible generator q is canonicalized to one
 * fixed hemisphere and an invisible generator -q is appended immediately. The
 * public mesh identifies q and -q and likewise keeps only one canonical member
 * of every antipodal vertex pair. This represents S^(d-1)/{x ~ -x}; in d=4 it
 * is the usual unit-quaternion representation of SO(3).
 */

#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace highvoronoi {

/** @brief Compile-time topology of the public spherical representation. */
enum class SphericalVoronoiMode {
    Sphere,
    AntipodalHemisphere
};

namespace detail {

/**
 * @brief Database guard enforcing the invariant "only the origin cell exists".
 *
 * The wrapped database still owns the central persistent identity hash and all
 * vertex payloads. Normal finite vertices are accepted only when their stable
 * internal signature contains node 0. Infinite edges are rejected immediately:
 * a valid spherical construction requires the Euclidean origin cell to be
 * bounded.
 */
template <class DatabaseT>
class SphericalCenterDataBase final {
public:
    using Database = DatabaseT;
    using Parameters = typename Database::Parameters;
    using Scalar = typename Database::Scalar;
    using Index = typename Database::Index;
    using Sigma = typename Database::Sigma;
    using VertexPoint = typename Database::VertexPoint;
    using LockType = typename Database::LockType;
    using address_type = std::size_t;

    static constexpr int DimensionAtCompileTime =
        Database::DimensionAtCompileTime;

    explicit SphericalCenterDataBase(std::shared_ptr<Database> database)
        : database_(require_database(std::move(database))) {}

    [[nodiscard]] address_type push(
        const VertexPoint& position,
        const Sigma& sigma) {
        validate_center_vertex(position, sigma);
        return database_->push(position, sigma);
    }

    void read(
        address_type address,
        VertexPoint& position,
        Sigma& sigma) const {
        database_->read(address, position, sigma);
    }

    [[nodiscard]] bool contains(const Sigma& sigma) const {
        return database_->contains(sigma);
    }

    bool erase(address_type address, const Sigma& sigma) {
        return database_->erase(address, sigma);
    }

    [[nodiscard]] bool register_signature(const Sigma& sigma) {
        return database_->register_signature(sigma);
    }

    [[nodiscard]] bool erase_signature(const Sigma& sigma) {
        return database_->erase_signature(sigma);
    }

    [[nodiscard]] address_type push_facet(
        const VertexPoint&,
        Sigma& sigma,
        const VertexPoint&) {
        // Computing the origin cell may transiently discover unbounded edges
        // belonging only to neighbouring outer cells. They are irrelevant for
        // the spherical mesh and must not be persisted. An unbounded edge that
        // actually contains node 0, however, means that the origin cell itself
        // is open and radial spherical projection is not a complete mesh.
        if (std::find(sigma.begin(), sigma.end(), Index{0}) != sigma.end()) {
            throw std::runtime_error(
                "SphericalVoronoiMesh origin cell is unbounded. "
                "The supplied spherical generators do not close the center cell.");
        }
        return address_type{0};
    }

    void read_facet(
        address_type address,
        VertexPoint& position,
        Sigma& sigma,
        VertexPoint& direction) const {
        database_->read_facet(address, position, sigma, direction);
    }

    [[nodiscard]] const std::shared_ptr<Database>&
    underlying_handle() const noexcept {
        return database_;
    }

private:
    [[nodiscard]] static std::shared_ptr<Database>
    require_database(std::shared_ptr<Database> database) {
        if (!database) {
            throw std::invalid_argument(
                "SphericalCenterDataBase requires a non-null database.");
        }
        return database;
    }

    static void validate_center_vertex(
        const VertexPoint& position,
        const Sigma& sigma) {
        if (sigma.empty() ||
            std::find(sigma.begin(), sigma.end(), Index{0}) == sigma.end()) {
            throw std::logic_error(
                "SphericalVoronoiMesh attempted to store a vertex that does "
                "not belong to the internal origin cell.");
        }

        long double norm_squared = 0.0L;
        for (std::size_t coordinate = 0;
             coordinate < static_cast<std::size_t>(position.size());
             ++coordinate) {
            const long double value = static_cast<long double>(
                position[static_cast<Eigen::Index>(coordinate)]);
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "SphericalVoronoiMesh produced a non-finite center-cell vertex.");
            }
            norm_squared += value * value;
        }

        if (!(norm_squared > 0.0L) || !std::isfinite(norm_squared)) {
            throw std::runtime_error(
                "SphericalVoronoiMesh produced the origin as a center-cell vertex; "
                "radial projection is undefined.");
        }
    }

    std::shared_ptr<Database> database_;
};

template <SphericalVoronoiMode Mode, class Index>
class SphericalModeState {
public:
    void initialize_origin() noexcept {}
    void reserve(std::size_t) {}
    void append_visible(Index) {}
    void append_antipode(Index, Index) {}

    [[nodiscard]] Index reference(Index internal) const noexcept {
        return internal;
    }
};

template <class Index>
class SphericalModeState<
    SphericalVoronoiMode::AntipodalHemisphere,
    Index> {
public:
    static constexpr Index invalid_index =
        (std::numeric_limits<Index>::max)();

    void initialize_origin() {
        reference_.assign(1, Index{0});
    }

    void reserve(std::size_t count) {
        reference_.reserve(count);
    }

    void append_visible(Index internal) {
        ensure_next(internal);
        reference_.push_back(internal);
    }

    void append_antipode(Index internal, Index visible_reference) {
        ensure_next(internal);
        reference_.push_back(visible_reference);
    }

    [[nodiscard]] Index reference(Index internal) const {
        const std::size_t position = static_cast<std::size_t>(internal);
        if (position >= reference_.size()) {
            throw std::out_of_range(
                "Spherical antipodal reference index out of range.");
        }
        return reference_[position];
    }

private:
    void ensure_next(Index internal) const {
        if (static_cast<std::size_t>(internal) != reference_.size()) {
            throw std::logic_error(
                "Spherical antipodal references require append-only stable indices.");
        }
    }

    std::vector<Index> reference_;
};

} // namespace detail

/**
 * @brief Voronoi mesh on S^(d-1), optionally modulo antipodal identification.
 *
 * @tparam NodeScalarT Scalar of public spherical generators.
 * @tparam Dim Ambient Euclidean dimension. The public sphere is S^(Dim-1).
 * @tparam DatabaseT Persistent finite-vertex database backend.
 * @tparam Mode Full sphere or one canonical antipodal hemisphere.
 */
template <typename NodeScalarT,
          int Dim,
          class DatabaseT,
          SphericalVoronoiMode Mode = SphericalVoronoiMode::Sphere>
class SphericalVoronoiMesh final {
public:
    using NodeScalar = NodeScalarT;
    using Database = DatabaseT;
    using VertexScalar = typename Database::Scalar;
    using Index = typename Database::Index;
    using Address = std::size_t;
    using NodePoint = PointType<NodeScalar, Dim>;
    using VertexPoint = PointType<VertexScalar, Dim>;
    using Sigma = std::vector<Index>;

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr SphericalVoronoiMode SphericalMode = Mode;
    static constexpr bool IsAntipodal =
        Mode == SphericalVoronoiMode::AntipodalHemisphere;

    using CenterDatabase = detail::SphericalCenterDataBase<Database>;
    using InternalMesh = VoronoiMesh<NodeScalar, Dim, CenterDatabase>;
    using InternalNodes = typename InternalMesh::InternalNodes;

    struct VertexRecord {
        Sigma sigma;
        VertexPoint position;
        Address address = Address{0};
    };

    struct RefineReport {
        std::size_t appended_public_nodes = 0;
        std::size_t appended_internal_nodes = 0;
        std::size_t invalidated_vertices = 0;
    };

    struct RemoveReport {
        std::size_t removed_public_nodes = 0;
        std::size_t removed_internal_nodes = 0;
    };

    struct Settings {
        /** Equality also invalidates a vertex because its support sigma changes. */
        VertexScalar nearest_tolerance = VertexScalar{1e-10};
    };

private:
    class PublicNodes final
        : public AbstractVoronoiNodes<NodeScalar, Dim, Index> {
    public:
        using Base = AbstractVoronoiNodes<NodeScalar, Dim, Index>;

        PublicNodes(const SphericalVoronoiMesh& owner, Index runtime_dimension)
            : Base(runtime_dimension), owner_(owner) {}

        [[nodiscard]] Index size() const noexcept override {
            return owner_.size();
        }

        [[nodiscard]] NodeScalar get_data(
            Index public_node,
            Index coordinate) const override {
            owner_.require_public_node(public_node);
            const Index internal = owner_.public_to_internal_.at(
                static_cast<std::size_t>(public_node));
            return owner_.internal_mesh_.internal_nodes().get_data(
                internal,
                coordinate);
        }

    private:
        void copy_node_impl(Index public_node, NodeScalar* target) const override {
            owner_.require_public_node(public_node);
            const Index internal = owner_.public_to_internal_.at(
                static_cast<std::size_t>(public_node));
            owner_.internal_mesh_.internal_nodes().copy_node(internal, target);
        }

        const SphericalVoronoiMesh& owner_;
    };

public:
    /** Construct an empty spherical mesh containing only internal origin node 0. */
    SphericalVoronoiMesh(
        Index runtime_dimension,
        std::shared_ptr<Database> database)
        : database_(require_database(std::move(database))),
          center_database_(
              std::make_shared<CenterDatabase>(database_)),
          internal_mesh_(
              make_origin_nodes(runtime_dimension),
              center_database_),
          public_nodes_(*this, checked_runtime_dimension(runtime_dimension)) {
        mode_state_.initialize_origin();
        internal_to_public_.assign(1, invalid_index());
    }

    /** Fixed-dimensional convenience constructor. */
    template <int D = Dim, std::enable_if_t<D != Dynamic, int> = 0>
    explicit SphericalVoronoiMesh(std::shared_ptr<Database> database)
        : SphericalVoronoiMesh(
              static_cast<Index>(Dim),
              std::move(database)) {}

    /** Construct and append the initial user-visible generator set. */
    SphericalVoronoiMesh(
        Index runtime_dimension,
        std::vector<NodePoint> nodes,
        std::shared_ptr<Database> database)
        : SphericalVoronoiMesh(
              runtime_dimension,
              std::move(database)) {
        (void)append_nodes(std::move(nodes));
    }

    /** Fixed-dimensional convenience constructor with initial nodes. */
    template <int D = Dim, std::enable_if_t<D != Dynamic, int> = 0>
    SphericalVoronoiMesh(
        std::vector<NodePoint> nodes,
        std::shared_ptr<Database> database)
        : SphericalVoronoiMesh(
              static_cast<Index>(Dim),
              std::move(nodes),
              std::move(database)) {}

    SphericalVoronoiMesh(const SphericalVoronoiMesh&) = delete;
    SphericalVoronoiMesh& operator=(const SphericalVoronoiMesh&) = delete;
    SphericalVoronoiMesh(SphericalVoronoiMesh&&) = delete;
    SphericalVoronoiMesh& operator=(SphericalVoronoiMesh&&) = delete;

    [[nodiscard]] Index dimension() const noexcept {
        return internal_mesh_.dimension();
    }

    [[nodiscard]] Index size() const noexcept {
        return static_cast<Index>(public_to_internal_.size());
    }

    [[nodiscard]] bool empty() const noexcept {
        return public_to_internal_.empty();
    }

    /** User-visible normalized/canonical spherical generators. */
    [[nodiscard]] const AbstractVoronoiNodes<NodeScalar, Dim, Index>&
    nodes() const noexcept {
        return public_nodes_;
    }

    [[nodiscard]] NodePoint node(Index public_node) const {
        return public_nodes_[public_node];
    }

    /** Stable internal Euclidean construction mesh, primarily for diagnostics. */
    [[nodiscard]] const InternalMesh& construction_mesh() const noexcept {
        return internal_mesh_;
    }

    /** Underlying user-provided persistent database. */
    [[nodiscard]] const std::shared_ptr<Database>&
    database_handle() const noexcept {
        return database_;
    }

    [[nodiscard]] Index public_node_to_internal(Index public_node) const {
        require_public_node(public_node);
        return public_to_internal_[static_cast<std::size_t>(public_node)];
    }

    [[nodiscard]] std::optional<Index>
    internal_node_to_public(Index internal_node) const {
        const std::size_t position = static_cast<std::size_t>(internal_node);
        if (position >= internal_to_public_.size()) {
            throw std::out_of_range(
                "SphericalVoronoiMesh internal node index out of range.");
        }
        const Index public_node = internal_to_public_[position];
        if (public_node == invalid_index()) {
            return std::nullopt;
        }
        return public_node;
    }

    /** In antipodal mode map an invisible stable node to its visible master. */
    [[nodiscard]] Index reference_internal_node(Index internal_node) const {
        if constexpr (IsAntipodal) {
            return mode_state_.reference(internal_node);
        } else {
            return internal_node;
        }
    }

    /** Append initial nodes. After compute(), use refine() instead. */
    [[nodiscard]] std::vector<Index> append_nodes(std::vector<NodePoint> nodes) {
        if (computed_) {
            throw std::logic_error(
                "SphericalVoronoiMesh::append_nodes() is only for initial setup; "
                "use refine() after compute().");
        }
        return append_visible_nodes(std::move(nodes)).public_indices;
    }

    [[nodiscard]] Index append_node(NodePoint node) {
        std::vector<NodePoint> nodes;
        nodes.push_back(std::move(node));
        return append_nodes(std::move(nodes)).front();
    }

    /** Compute the complete internal origin cell exactly once. */
    template <
        class SearchKeyword = geometry::KDSearch,
        class RayParameters = RaycastParameters<CombinedRaycast, VertexScalar>,
        class MeshThreading = SingleThread,
        class CastThreading = SingleThread>
    void compute(
        SearchKeyword search_keyword = SearchKeyword{},
        RayParameters ray_parameters = RayParameters{},
        MeshThreading mesh_threading = MeshThreading{},
        CastThreading cast_threading = CastThreading{}) {
        if (computed_) {
            throw std::logic_error(
                "SphericalVoronoiMesh::compute() may only be called once; "
                "use refine()/remove() afterwards.");
        }
        run_center_compute(
            search_keyword,
            ray_parameters,
            mesh_threading,
            cast_threading);
        computed_ = true;
    }

    /**
     * Append visible generators (and antipodes), invalidate obsolete center
     * vertices, then run the unchanged Euclidean ComputeVoronoi on cell 0.
     */
    template <
        class SearchKeyword = geometry::KDSearch,
        class RayParameters = RaycastParameters<CombinedRaycast, VertexScalar>,
        class MeshThreading = SingleThread,
        class CastThreading = SingleThread>
    [[nodiscard]] RefineReport refine(
        std::vector<NodePoint> nodes,
        SearchKeyword search_keyword = SearchKeyword{},
        RayParameters ray_parameters = RayParameters{},
        MeshThreading mesh_threading = MeshThreading{},
        CastThreading cast_threading = CastThreading{},
        Settings settings = Settings{}) {
        require_computed("refine");
        if (nodes.empty()) {
            return {};
        }

        AppendResult appended = append_visible_nodes(std::move(nodes));

        RefineReport report;
        report.appended_public_nodes = appended.public_indices.size();
        report.appended_internal_nodes = appended.internal_indices.size();

        const std::vector<NodePoint> new_internal_points =
            materialize_internal_nodes(appended.internal_indices);

        report.invalidated_vertices = internal_mesh_.erase_vertices_if(
            [&](const Sigma&, const VertexPoint& vertex) {
                const long double center_distance = squared_norm(vertex);
                for (const NodePoint& node : new_internal_points) {
                    const long double candidate_distance =
                        squared_distance(vertex, node);
                    const long double tolerance = static_cast<long double>(
                        settings.nearest_tolerance);
                    if (candidate_distance <= center_distance + tolerance) {
                        return true;
                    }
                }
                return false;
            });

        run_center_compute(
            search_keyword,
            ray_parameters,
            mesh_threading,
            cast_threading);
        return report;
    }

    /** Remove public generators (and antipodes) and reconstruct the origin cell. */
    template <
        class SearchKeyword = geometry::KDSearch,
        class RayParameters = RaycastParameters<CombinedRaycast, VertexScalar>,
        class MeshThreading = SingleThread,
        class CastThreading = SingleThread>
    [[nodiscard]] RemoveReport remove(
        std::vector<Index> deleted_public_nodes,
        SearchKeyword search_keyword = SearchKeyword{},
        RayParameters ray_parameters = RayParameters{},
        MeshThreading mesh_threading = MeshThreading{},
        CastThreading cast_threading = CastThreading{}) {
        require_computed("remove");
        if (deleted_public_nodes.empty()) {
            return {};
        }

        std::sort(deleted_public_nodes.begin(), deleted_public_nodes.end());
        if (std::adjacent_find(
                deleted_public_nodes.begin(),
                deleted_public_nodes.end()) != deleted_public_nodes.end()) {
            throw std::invalid_argument(
                "SphericalVoronoiMesh::remove() received duplicate public nodes.");
        }
        for (const Index public_node : deleted_public_nodes) {
            require_public_node(public_node);
        }

        std::vector<Index> deleted_internal_nodes;
        deleted_internal_nodes.reserve(
            deleted_public_nodes.size() * (IsAntipodal ? 2u : 1u));

        for (const Index public_node : deleted_public_nodes) {
            const Index visible_internal =
                public_to_internal_[static_cast<std::size_t>(public_node)];
            deleted_internal_nodes.push_back(visible_internal);
            if constexpr (IsAntipodal) {
                deleted_internal_nodes.push_back(
                    find_antipode_internal(visible_internal));
            }
        }
        std::sort(deleted_internal_nodes.begin(), deleted_internal_nodes.end());

        const std::size_t removed_internal = internal_mesh_.erase_nodes_if(
            [&](Index internal_public, const NodePoint&) {
                const Index stable_internal =
                    internal_mesh_.public_node_to_internal(internal_public);
                return std::binary_search(
                    deleted_internal_nodes.begin(),
                    deleted_internal_nodes.end(),
                    stable_internal);
            });

        for (auto it = deleted_public_nodes.rbegin();
             it != deleted_public_nodes.rend();
             ++it) {
            public_to_internal_.erase(
                public_to_internal_.begin() +
                static_cast<std::ptrdiff_t>(*it));
        }
        rebuild_internal_to_public();

        run_center_compute(
            search_keyword,
            ray_parameters,
            mesh_threading,
            cast_threading);

        return RemoveReport{
            deleted_public_nodes.size(),
            removed_internal};
    }

    // ---------------------------------------------------------------------
    // Spherical neighbour interface
    // ---------------------------------------------------------------------

    /**
     * @brief Load the last stored neighbour list of one public spherical cell.
     *
     * Neighbour geometry is computed unchanged on the corresponding ordinary
     * Euclidean cell of the internal origin-augmented mesh.  That internal
     * record deliberately retains node 0: C_0 is a genuine Euclidean facet
     * neighbour of every spherical generator and keeping it preserves the
     * complete internal topology for later algorithms.
     *
     * The public spherical projection removes only internal origin node 0.
     * In antipodal mode an internal neighbour q or -q is projected to the same
     * visible master.  Multiplicity is intentionally preserved: two distinct
     * quotient interfaces may therefore appear as the same public neighbour
     * index more than once.
     *
     * @return Exactly `!dirty(public_cell)`.  A stale stored record, when
     *         present, is still projected into `buffer`.
     */
    [[nodiscard]] bool neighbours(
        Index public_cell,
        std::vector<Index>& buffer) const {
        require_computed("neighbours");
        require_public_node(public_cell);

        const Index visible_internal =
            public_to_internal_[static_cast<std::size_t>(public_cell)];
        (void)internal_mesh_.internal_neighbours(visible_internal, buffer);
        project_internal_neighbours(buffer);
        return !dirty(public_cell);
    }

    /** @brief Return whether one public spherical neighbour list is stale. */
    [[nodiscard]] bool dirty(Index public_cell) const {
        require_computed("dirty");
        require_public_node(public_cell);

        const Index visible_internal =
            public_to_internal_[static_cast<std::size_t>(public_cell)];
        bool result =
            internal_mesh_.internal_neighbours_dirty(visible_internal);

        if constexpr (IsAntipodal) {
            result = result || internal_mesh_.internal_neighbours_dirty(
                find_antipode_internal(visible_internal));
        }
        return result;
    }

    /**
     * @brief Set or clear the neighbour dirty state of one public cell.
     *
     * In antipodal mode both internal representatives belong to the same
     * public quotient cell and therefore receive the same dirty state.
     */
    void set_dirty(Index public_cell, bool value = true) {
        require_computed("set_dirty");
        require_public_node(public_cell);

        const Index visible_internal =
            public_to_internal_[static_cast<std::size_t>(public_cell)];
        internal_mesh_.set_internal_neighbours_dirty(visible_internal, value);

        if constexpr (IsAntipodal) {
            internal_mesh_.set_internal_neighbours_dirty(
                find_antipode_internal(visible_internal),
                value);
        }
    }

    /**
     * @brief Recompute one public spherical cell's neighbours.
     *
     * No spherical special-case geometry is used.  The existing ordinary
     * NeighbourFinder is run on the corresponding stable internal Euclidean
     * cell.  Its stored result includes internal origin node 0.  Only the
     * public `neighbours()` projection removes that origin.
     *
     * Antipodal quotient cells have two internal Euclidean representatives;
     * both records are computed so the internal mesh remains topologically
     * complete even though public presentation uses the visible representative
     * only.
     */
    void compute_neighbors(Index public_cell) {
        require_computed("compute_neighbors");
        require_public_node(public_cell);

        const Index visible_internal =
            public_to_internal_[static_cast<std::size_t>(public_cell)];
        internal_mesh_.compute_internal_neighbours(visible_internal);

        if constexpr (IsAntipodal) {
            internal_mesh_.compute_internal_neighbours(
                find_antipode_internal(visible_internal));
        }
    }

    /** Return every public spherical Voronoi vertex exactly once. */
    [[nodiscard]] std::vector<VertexRecord> vertices() const {
        require_computed("vertices");
        std::vector<VertexRecord> result;
        for (const auto& internal_vertex : internal_mesh_.primary_vertices(Index{0})) {
            append_public_vertex(internal_vertex, result);
        }
        return result;
    }

    /** Return the public spherical vertices incident to one visible generator. */
    [[nodiscard]] std::vector<VertexRecord> vertices(Index public_node) const {
        require_computed("vertices");
        require_public_node(public_node);

        std::vector<VertexRecord> result;
        const Index visible_internal =
            public_to_internal_[static_cast<std::size_t>(public_node)];
        append_secondary_vertices(visible_internal, result);

        if constexpr (IsAntipodal) {
            append_secondary_vertices(
                find_antipode_internal(visible_internal),
                result);
        }
        return result;
    }

private:
    struct AppendResult {
        std::vector<Index> public_indices;
        std::vector<Index> internal_indices;
    };

    [[nodiscard]] static constexpr Index invalid_index() noexcept {
        return (std::numeric_limits<Index>::max)();
    }

    [[nodiscard]] static std::shared_ptr<Database>
    require_database(std::shared_ptr<Database> database) {
        if (!database) {
            throw std::invalid_argument(
                "SphericalVoronoiMesh requires a non-null database.");
        }
        return database;
    }

    [[nodiscard]] static Index checked_runtime_dimension(Index runtime_dimension) {
        if constexpr (Dim == Dynamic) {
            if (runtime_dimension == Index{0}) {
                throw std::invalid_argument(
                    "Dynamic SphericalVoronoiMesh dimension must be positive.");
            }
            return runtime_dimension;
        } else {
            if (runtime_dimension != static_cast<Index>(Dim)) {
                throw std::invalid_argument(
                    "SphericalVoronoiMesh runtime dimension does not match Dim.");
            }
            return static_cast<Index>(Dim);
        }
    }

    [[nodiscard]] static InternalNodes make_origin_nodes(Index runtime_dimension) {
        const Index checked = checked_runtime_dimension(runtime_dimension);
        InternalNodes nodes = [&]() {
            if constexpr (Dim == Dynamic) {
                return InternalNodes(Index{1}, checked);
            } else {
                return InternalNodes(Index{1});
            }
        }();
        NodePoint origin = make_node_point(checked);
        origin.setZero();
        nodes.set(Index{0}, origin);
        return nodes;
    }

    [[nodiscard]] static NodePoint make_node_point(Index runtime_dimension) {
        if constexpr (Dim == Dynamic) {
            return NodePoint(static_cast<Eigen::Index>(runtime_dimension));
        } else {
            (void)runtime_dimension;
            return NodePoint{};
        }
    }

    template <class PointLike>
    [[nodiscard]] static bool canonical_hemisphere(const PointLike& point) {
        for (Eigen::Index coordinate = 0;
             coordinate < point.size();
             ++coordinate) {
            const auto value = point[coordinate];
            if (value > decltype(value){0}) {
                return true;
            }
            if (value < decltype(value){0}) {
                return false;
            }
        }
        return true;
    }

    void normalize_and_canonicalize(NodePoint& point) const {
        if (static_cast<Index>(point.size()) != dimension()) {
            throw std::invalid_argument(
                "SphericalVoronoiMesh node dimension mismatch.");
        }

        long double norm_squared = 0.0L;
        for (Eigen::Index coordinate = 0;
             coordinate < point.size();
             ++coordinate) {
            const long double value =
                static_cast<long double>(point[coordinate]);
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "SphericalVoronoiMesh nodes must be finite.");
            }
            norm_squared += value * value;
        }
        if (!(norm_squared > 0.0L) || !std::isfinite(norm_squared)) {
            throw std::invalid_argument(
                "SphericalVoronoiMesh nodes must be non-zero.");
        }

        const long double norm = std::sqrt(norm_squared);
        for (Eigen::Index coordinate = 0;
             coordinate < point.size();
             ++coordinate) {
            point[coordinate] = static_cast<NodeScalar>(
                static_cast<long double>(point[coordinate]) / norm);
        }

        if constexpr (IsAntipodal) {
            if (!canonical_hemisphere(point)) {
                point = -point;
            }
        }
    }

    [[nodiscard]] AppendResult append_visible_nodes(std::vector<NodePoint> nodes) {
        if (nodes.empty()) {
            return {};
        }

        for (NodePoint& point : nodes) {
            normalize_and_canonicalize(point);
        }

        std::vector<NodePoint> internal_points;
        internal_points.reserve(nodes.size() * (IsAntipodal ? 2u : 1u));
        for (const NodePoint& point : nodes) {
            internal_points.push_back(point);
            if constexpr (IsAntipodal) {
                internal_points.push_back(-point);
            }
        }

        const std::vector<Index> stable_internal =
            internal_mesh_.append_nodes(internal_points);
        if (stable_internal.size() != internal_points.size()) {
            throw std::logic_error(
                "SphericalVoronoiMesh internal append size mismatch.");
        }

        AppendResult result;
        result.public_indices.reserve(nodes.size());
        result.internal_indices = stable_internal;

        internal_to_public_.resize(
            static_cast<std::size_t>(internal_mesh_.internal_nodes().size()),
            invalid_index());
        mode_state_.reserve(
            static_cast<std::size_t>(internal_mesh_.internal_nodes().size()));

        std::size_t internal_position = 0;
        for (std::size_t public_offset = 0;
             public_offset < nodes.size();
             ++public_offset) {
            const Index public_node = static_cast<Index>(public_to_internal_.size());
            const Index visible_internal = stable_internal[internal_position++];

            public_to_internal_.push_back(visible_internal);
            internal_to_public_[static_cast<std::size_t>(visible_internal)] =
                public_node;
            mode_state_.append_visible(visible_internal);
            result.public_indices.push_back(public_node);

            if constexpr (IsAntipodal) {
                const Index antipode_internal = stable_internal[internal_position++];
                mode_state_.append_antipode(
                    antipode_internal,
                    visible_internal);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<NodePoint> materialize_internal_nodes(
        const std::vector<Index>& internal_indices) const {
        std::vector<NodePoint> result;
        result.reserve(internal_indices.size());
        for (const Index internal : internal_indices) {
            result.push_back(internal_mesh_.internal_nodes()[internal]);
        }
        return result;
    }

    template <class A, class B>
    [[nodiscard]] static long double squared_distance(
        const A& a,
        const B& b) {
        long double result = 0.0L;
        for (Eigen::Index coordinate = 0;
             coordinate < a.size();
             ++coordinate) {
            const long double difference =
                static_cast<long double>(a[coordinate]) -
                static_cast<long double>(b[coordinate]);
            result += difference * difference;
        }
        return result;
    }

    template <class A>
    [[nodiscard]] static long double squared_norm(const A& a) {
        long double result = 0.0L;
        for (Eigen::Index coordinate = 0;
             coordinate < a.size();
             ++coordinate) {
            const long double value =
                static_cast<long double>(a[coordinate]);
            result += value * value;
        }
        return result;
    }

    template <class SearchKeyword,
              class RayParameters,
              class MeshThreading,
              class CastThreading>
    void run_center_compute(
        const SearchKeyword& search_keyword,
        const RayParameters& ray_parameters,
        MeshThreading mesh_threading,
        CastThreading cast_threading) {
        if (size() == Index{0}) {
            throw std::logic_error(
                "SphericalVoronoiMesh requires at least one visible generator.");
        }

        auto tree = geometry::make_search_tree(
            internal_mesh_,
            search_keyword);
        auto raycaster = make_raycaster(tree, ray_parameters);

        using RayCaster = decltype(raycaster);
        using Compute = ComputeVoronoi<
            InternalMesh,
            RayCaster,
            MeshThreading,
            CastThreading>;

        Compute compute(
            internal_mesh_,
            raycaster,
            std::move(mesh_threading),
            std::move(cast_threading),
            Index{1});
        compute.compute();
    }

    void require_public_node(Index public_node) const {
        if (public_node >= size()) {
            throw std::out_of_range(
                "SphericalVoronoiMesh public node index out of range.");
        }
    }

    void require_computed(const char* operation) const {
        if (!computed_) {
            throw std::logic_error(
                std::string("SphericalVoronoiMesh::") + operation +
                "() requires an initial compute().");
        }
    }

    void rebuild_internal_to_public() {
        std::fill(
            internal_to_public_.begin(),
            internal_to_public_.end(),
            invalid_index());
        for (Index public_node = Index{0};
             public_node < size();
             ++public_node) {
            const Index internal =
                public_to_internal_[static_cast<std::size_t>(public_node)];
            internal_to_public_[static_cast<std::size_t>(internal)] = public_node;
        }
    }

    [[nodiscard]] Index find_antipode_internal(Index visible_internal) const {
        // Visible and antipodal nodes are appended consecutively. Stable slots
        // are never recycled, so this relation remains valid after deletions.
        const Index candidate = static_cast<Index>(visible_internal + Index{1});
        if (candidate >= internal_mesh_.internal_nodes().size() ||
            mode_state_.reference(candidate) != visible_internal) {
            throw std::logic_error(
                "SphericalVoronoiMesh lost an antipodal node reference.");
        }
        return candidate;
    }

    template <class InternalVertexRecord>
    void append_public_vertex(
        const InternalVertexRecord& internal_vertex,
        std::vector<VertexRecord>& result) const {
        VertexPoint projected = internal_vertex.position;
        const VertexScalar norm = projected.norm();
        if (!(norm > VertexScalar{0}) || !std::isfinite(norm)) {
            throw std::runtime_error(
                "SphericalVoronoiMesh cannot project an invalid center-cell vertex.");
        }
        projected /= norm;

        if constexpr (IsAntipodal) {
            if (!canonical_hemisphere(projected)) {
                return;
            }
        }

        Sigma public_sigma;
        public_sigma.reserve(internal_vertex.sigma.size());

        for (const Index internal_public : internal_vertex.sigma) {
            const Index stable_internal =
                internal_mesh_.public_node_to_internal(internal_public);
            if (stable_internal == Index{0}) {
                continue;
            }

            Index visible_internal = stable_internal;
            if constexpr (IsAntipodal) {
                visible_internal = mode_state_.reference(stable_internal);
            }

            const std::size_t position =
                static_cast<std::size_t>(visible_internal);
            if (position >= internal_to_public_.size()) {
                throw std::logic_error(
                    "SphericalVoronoiMesh vertex references an unknown internal node.");
            }
            const Index public_node = internal_to_public_[position];
            if (public_node == invalid_index()) {
                throw std::logic_error(
                    "SphericalVoronoiMesh vertex references an inactive visible node.");
            }
            public_sigma.push_back(public_node);
        }

        std::sort(public_sigma.begin(), public_sigma.end());
        public_sigma.erase(
            std::unique(public_sigma.begin(), public_sigma.end()),
            public_sigma.end());

        if (public_sigma.empty()) {
            throw std::logic_error(
                "SphericalVoronoiMesh projected an empty public vertex signature.");
        }

        result.push_back(VertexRecord{
            std::move(public_sigma),
            std::move(projected),
            internal_vertex.address});
    }

    /** Project one stable-internal neighbour record to spherical public form. */
    void project_internal_neighbours(std::vector<Index>& neighbours) const {
        std::size_t write = 0;
        const Index internal_count = internal_mesh_.internal_nodes().size();

        for (std::size_t read = 0; read < neighbours.size(); ++read) {
            const Index internal_neighbour = neighbours[read];

            // Internal origin is a genuine Euclidean neighbour and remains in
            // the stored internal record.  It has no spherical public cell.
            if (internal_neighbour == Index{0}) {
                continue;
            }
            if (internal_neighbour >= internal_count) {
                throw std::logic_error(
                    "SphericalVoronoiMesh internal neighbour record contains "
                    "a non-ordinary generator.");
            }

            Index visible_internal = internal_neighbour;
            if constexpr (IsAntipodal) {
                visible_internal = mode_state_.reference(internal_neighbour);
            }

            const std::size_t position =
                static_cast<std::size_t>(visible_internal);
            if (position >= internal_to_public_.size()) {
                throw std::logic_error(
                    "SphericalVoronoiMesh neighbour references an unknown "
                    "stable internal node.");
            }

            const Index public_neighbour = internal_to_public_[position];
            if (public_neighbour == invalid_index()) {
                // A stale historical neighbour record may still reference a
                // deleted visible generator.  Match AbstractMesh::neighbours()
                // and omit entries without a current public representation.
                continue;
            }

            neighbours[write++] = public_neighbour;
        }
        neighbours.resize(write);
    }

    void append_secondary_vertices(
        Index stable_internal,
        std::vector<VertexRecord>& result) const {
        const std::optional<Index> internal_public =
            internal_mesh_.internal_node_to_public(stable_internal);
        if (!internal_public) {
            return;
        }
        for (const auto& vertex :
             internal_mesh_.secondary_vertices(*internal_public)) {
            append_public_vertex(vertex, result);
        }
    }

    std::shared_ptr<Database> database_;
    std::shared_ptr<CenterDatabase> center_database_;
    InternalMesh internal_mesh_;
    detail::SphericalModeState<Mode, Index> mode_state_;
    std::vector<Index> public_to_internal_;
    std::vector<Index> internal_to_public_;
    PublicNodes public_nodes_;
    bool computed_ = false;
};

/** Convenient aliases for the two compile-time modes. */
template <typename NodeScalarT, int Dim, class DatabaseT>
using SphereVoronoiMesh = SphericalVoronoiMesh<
    NodeScalarT,
    Dim,
    DatabaseT,
    SphericalVoronoiMode::Sphere>;

template <typename NodeScalarT, int Dim, class DatabaseT>
using AntipodalSphericalVoronoiMesh = SphericalVoronoiMesh<
    NodeScalarT,
    Dim,
    DatabaseT,
    SphericalVoronoiMode::AntipodalHemisphere>;

} // namespace highvoronoi
