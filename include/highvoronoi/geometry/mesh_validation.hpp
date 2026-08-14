#pragma once

/**
 * @file mesh_validation.hpp
 * @brief Consistency checks and comparisons for HighVoronoi meshes.
 *
 * `verify_mesh()` checks the geometric consistency of every currently stored
 * vertex. It does not attempt to prove that a mesh is complete. For every
 * public cell it activates the same boundary-mirror representation used by the
 * Voronoi algorithm, walks the cell's visible vertices, evaluates the
 * RayCaster vertex variance, and checks that a nearest generator belongs to
 * the stored signature.
 *
 * `compare_meshes()` checks whether two compatible meshes contain the same
 * ordinary nodes and the same primary vertices. Vertex identity is determined
 * by the public signature. For matching signatures, positions are compared
 * with a tolerance derived from the scale-free vertex variance. The square
 * root of the maximum variance is multiplied by the local Voronoi radius to
 * obtain a length. A small floating-point roundoff floor is added so two valid
 * constructions that differ only by operation order are not reported as
 * geometrically different.
 *
 * Both functions return detailed reports. Mesh comparisons additionally
 * accumulate the sum, mean, and maximum Euclidean position difference over
 * all primary vertices with matching signatures. Diagnostic printing is
 * optional and is disabled by default.
 */

#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/edge_iterator.hpp>
#include <highvoronoi/detail/edge_hash_table.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

namespace highvoronoi {

/** Kind of inconsistency found by verify_mesh(). */
enum class MeshVerificationErrorKind {
    Variance,
    NearestNeighbor
};

template <class MeshT>
struct MeshVerificationError {
    using Mesh = MeshT;
    using Index = typename Mesh::Index;
    using Address = typename Mesh::Address;
    using Sigma = typename Mesh::Sigma;
    using Point = typename Mesh::VertexPoint;
    using Scalar = typename Mesh::NodeScalar;

    MeshVerificationErrorKind kind{};
    Index cell{};
    Address address{};
    Sigma sigma;
    Point position;
    Scalar variance{};
    std::optional<Index> nearest_neighbor;
};

template <class MeshT>
struct MeshVerificationReport {
    using Mesh = MeshT;
    using Error = MeshVerificationError<Mesh>;

    std::size_t checked_vertex_occurrences{0};
    std::vector<Error> errors;

    [[nodiscard]] bool valid() const noexcept {
        return errors.empty();
    }

    [[nodiscard]] std::size_t error_count() const noexcept {
        return errors.size();
    }
};

/**
 * @brief Topological completeness report based on full-edge endpoint counting.
 *
 * Every geometric finite edge must occur once at each endpoint. An unbounded
 * edge occurs once at its finite endpoint and once in the persisted infinite-
 * edge list. Degenerate minimal-edge representations are deduplicated by their
 * complete supporting edge before the global occurrence count is updated.
 */
template <class MeshT>
struct MeshCompletenessReport {
    using Mesh = MeshT;

    MeshVerificationReport<Mesh> consistency;
    std::size_t unique_finite_edge_endpoints{0};
    std::size_t duplicate_local_edge_representations{0};
    std::size_t infinite_edges{0};
    bool all_edges_have_two_occurrences{false};

    [[nodiscard]] bool complete() const noexcept {
        return consistency.valid() && all_edges_have_two_occurrences;
    }
};

/** Kind of difference found by compare_meshes(). */
enum class MeshComparisonErrorKind {
    DimensionMismatch,
    NodeCountMismatch,
    NodePositionMismatch,
    MissingInFirst,
    MissingInSecond,
    VertexPositionMismatch
};

template <class MeshT>
struct MeshComparisonError {
    using Mesh = MeshT;
    using Index = typename Mesh::Index;
    using Sigma = typename Mesh::Sigma;
    using Point = typename Mesh::VertexPoint;
    using Scalar = typename Mesh::NodeScalar;

    MeshComparisonErrorKind kind{};
    Index cell{};
    Sigma sigma;
    Point first_position;
    Point second_position;
    Scalar distance{};
    Scalar tolerance{};
};

template <class MeshT>
struct MeshComparisonReport {
    using Mesh = MeshT;
    using Error = MeshComparisonError<Mesh>;
    using Scalar = typename Mesh::NodeScalar;

    std::size_t checked_nodes{0};
    std::size_t checked_primary_vertices{0};
    std::size_t matched_primary_vertices{0};
    Scalar vertex_distance_sum{0};
    Scalar vertex_distance_max{0};
    std::vector<Error> errors;

    [[nodiscard]] bool equal() const noexcept {
        return errors.empty();
    }

    [[nodiscard]] std::size_t error_count() const noexcept {
        return errors.size();
    }

    [[nodiscard]] Scalar mean_vertex_distance() const noexcept {
        return matched_primary_vertices == 0
            ? Scalar{0}
            : vertex_distance_sum /
                  static_cast<Scalar>(matched_primary_vertices);
    }

    void print_summary(std::ostream& output) const {
        output
            << "checked nodes:             " << checked_nodes << '\n'
            << "checked primary vertices:  " << checked_primary_vertices << '\n'
            << "matched primary vertices:  " << matched_primary_vertices << '\n'
            << "sum of position errors:    " << vertex_distance_sum << '\n'
            << "mean position error:       " << mean_vertex_distance() << '\n'
            << "max position error:        " << vertex_distance_max << '\n'
            << "comparison errors:         " << error_count() << '\n';
    }
};

namespace detail {

template <class Point>
[[nodiscard]] Point make_validation_point(std::size_t dimension) {
    if constexpr (Point::SizeAtCompileTime == Eigen::Dynamic) {
        return Point(static_cast<Eigen::Index>(dimension));
    } else {
        (void)dimension;
        return Point{};
    }
}

template <class Source, class Target>
void copy_validation_point(const Source& source, Target& target) {
    if (source.size() != target.size()) {
        throw std::invalid_argument(
            "Mesh-validation point dimensions do not match.");
    }

    for (Eigen::Index i = 0; i < source.size(); ++i) {
        target[i] = static_cast<typename Target::Scalar>(source[i]);
    }
}

template <class ExtendedNodes>
[[nodiscard]] auto all_mirror_indices(const ExtendedNodes& nodes) {
    using Index = typename ExtendedNodes::Index;

    std::vector<Index> result;
    result.reserve(static_cast<std::size_t>(nodes.mirror_count()));

    for (Index plane = Index{0}; plane < nodes.mirror_count(); ++plane) {
        result.push_back(nodes.mirror_index(plane));
    }
    return result;
}

template <class Sigma, class Index>
[[nodiscard]] bool sigma_contains(const Sigma& sigma, Index index) {
    return std::find(sigma.begin(), sigma.end(), index) != sigma.end();
}

template <class Sigma>
void print_sigma(std::ostream& output, const Sigma& sigma) {
    output << '(';
    for (std::size_t i = 0; i < static_cast<std::size_t>(sigma.size()); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << sigma[i];
    }
    output << ')';
}

template <class Point>
void print_point(std::ostream& output, const Point& point) {
    output << '(';
    for (Eigen::Index i = 0; i < point.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << point[i];
    }
    output << ')';
}

template <class Mesh>
using ValidationTree = decltype(
    geometry::make_search_tree(
        std::declval<Mesh&>(),
        geometry::KDSearch{}));

template <class Mesh>
using ValidationRayCaster = decltype(
    make_raycaster(
        std::declval<const ValidationTree<Mesh>&>(),
        RaycastParameters<ClassicRaycast, typename Mesh::NodeScalar>{}));

template <class Mesh>
class MeshValidationContext final {
public:
    using Scalar = typename Mesh::NodeScalar;
    using Index = typename Mesh::Index;
    using RayCaster = ValidationRayCaster<Mesh>;
    using Point = typename RayCaster::Point;

    explicit MeshValidationContext(Mesh& mesh)
        : tree_(geometry::make_search_tree(
              mesh,
              geometry::KDSearch{8, 1})),
          raycaster_(make_raycaster(
              tree_,
              RaycastParameters<ClassicRaycast, Scalar>{})),
          search_data_(raycaster_.tree().make_backend_data()),
          point_(make_validation_point<Point>(
              static_cast<std::size_t>(mesh.dimension()))),
          mirror_indices_(all_mirror_indices(
              raycaster_.extended_nodes())) {}

    void activate_cell(Index cell) {
        raycaster_.activate_cell(cell, mirror_indices_);
    }

    template <class Position>
    void load_position(const Position& position) {
        copy_validation_point(position, point_);
    }

    template <class Sigma>
    [[nodiscard]] Scalar variance(const Sigma& sigma) {
        return raycaster_.vertex_variance(sigma, point_);
    }

    [[nodiscard]] std::optional<Index> nearest_neighbor() {
        raycaster_.tree().write_point(search_data_, point_);
        const auto nearest = raycaster_.tree().nn(
            search_data_,
            [](Index) noexcept { return false; });
        if (!nearest) {
            return std::nullopt;
        }
        return nearest->index;
    }

    [[nodiscard]] const Point& point() const noexcept {
        return point_;
    }

    [[nodiscard]] auto& extended_nodes() noexcept {
        return raycaster_.extended_nodes();
    }

private:
    ValidationTree<Mesh> tree_;
    RayCaster raycaster_;
    typename RayCaster::SearchData search_data_;
    Point point_;
    std::vector<Index> mirror_indices_;
};

template <class Mesh>
void print_verification_error(
    std::ostream& output,
    const MeshVerificationError<Mesh>& error,
    typename Mesh::NodeScalar maximum_variance) {

    output << "[mesh verification] cell " << error.cell
           << ", address " << error.address << ": ";

    if (error.kind == MeshVerificationErrorKind::Variance) {
        output << "variance at sigma=";
        print_sigma(output, error.sigma);
        output << ", r=";
        print_point(output, error.position);
        output << " is " << error.variance
               << " > " << maximum_variance << '\n';
        return;
    }

    output << "nearest neighbor of r=";
    print_point(output, error.position);
    if (error.nearest_neighbor) {
        output << " is " << *error.nearest_neighbor
               << " but not in sigma=";
        print_sigma(output, error.sigma);
    } else {
        output << " was not found for sigma=";
        print_sigma(output, error.sigma);
    }
    output << '\n';
}

template <class Mesh>
[[nodiscard]] const typename Mesh::VertexRecord* find_primary_vertex(
    const Mesh& mesh,
    typename Mesh::Index cell,
    const typename Mesh::Sigma& sigma,
    typename Mesh::VertexRecord& storage) {

    for (const auto& vertex : mesh.primary_vertices(cell)) {
        if (vertex.sigma == sigma) {
            storage = vertex;
            return &storage;
        }
    }
    return nullptr;
}

template <class Mesh>
void print_comparison_error(
    std::ostream& output,
    const MeshComparisonError<Mesh>& error) {

    output << "[mesh comparison] ";

    switch (error.kind) {
    case MeshComparisonErrorKind::DimensionMismatch:
        output << "mesh dimensions differ";
        break;
    case MeshComparisonErrorKind::NodeCountMismatch:
        output << "node counts differ";
        break;
    case MeshComparisonErrorKind::NodePositionMismatch:
        output << "node " << error.cell << " positions differ by "
               << error.distance << " > " << error.tolerance;
        break;
    case MeshComparisonErrorKind::MissingInFirst:
        output << "vertex sigma=";
        print_sigma(output, error.sigma);
        output << " is missing in first mesh";
        break;
    case MeshComparisonErrorKind::MissingInSecond:
        output << "vertex sigma=";
        print_sigma(output, error.sigma);
        output << " is missing in second mesh";
        break;
    case MeshComparisonErrorKind::VertexPositionMismatch:
        output << "vertex sigma=";
        print_sigma(output, error.sigma);
        output << " positions differ by " << error.distance
               << " > tolerance=" << error.tolerance;
        break;
    }

    output << '\n';
}

} // namespace detail

/**
 * @brief Check geometric consistency of all currently stored mesh vertices.
 *
 * The function deliberately does not test completeness. A mesh with missing
 * vertices may therefore pass if every stored vertex is individually valid.
 *
 * Each vertex occurrence is checked in the context of the cell through which
 * it is iterated, so boundary mirrors are activated exactly for that cell.
 */
template <class Mesh>
[[nodiscard]] MeshVerificationReport<Mesh> verify_mesh(
    Mesh& mesh,
    typename Mesh::NodeScalar maximum_variance =
        typename Mesh::NodeScalar{1e-20},
    bool print_errors = false,
    std::ostream& output = std::cerr) {

    using Index = typename Mesh::Index;
    using Error = MeshVerificationError<Mesh>;

    MeshVerificationReport<Mesh> report;
    detail::MeshValidationContext<Mesh> context(mesh);

    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        context.activate_cell(cell);

        for (const auto& vertex : mesh.vertices(cell)) {
            ++report.checked_vertex_occurrences;
            context.load_position(vertex.position);

            const auto variance = context.variance(vertex.sigma);
            if (!std::isfinite(static_cast<long double>(variance)) ||
                variance > maximum_variance) {
                Error error{
                    MeshVerificationErrorKind::Variance,
                    cell,
                    vertex.address,
                    vertex.sigma,
                    vertex.position,
                    variance,
                    std::nullopt};
                if (print_errors) {
                    detail::print_verification_error(
                        output,
                        error,
                        maximum_variance);
                }
                report.errors.push_back(std::move(error));
            }

            const auto nearest = context.nearest_neighbor();
            if (!nearest ||
                !detail::sigma_contains(vertex.sigma, *nearest)) {
                Error error{
                    MeshVerificationErrorKind::NearestNeighbor,
                    cell,
                    vertex.address,
                    vertex.sigma,
                    vertex.position,
                    variance,
                    nearest};
                if (print_errors) {
                    detail::print_verification_error(
                        output,
                        error,
                        maximum_variance);
                }
                report.errors.push_back(std::move(error));
            }
        }
    }

    return report;
}

/**
 * @brief Check geometric consistency and global Voronoi-edge closure.
 *
 * The finite endpoint contribution of one geometric edge is identified by the
 * EdgeIterator's complete supporting edge (`full_indices()`), not by its local
 * minimal edge. This matters for degenerate vertices where the same geometric
 * edge may be represented by several minimal edges and cell perspectives. Such
 * repetitions are collapsed per persistent vertex address before the global
 * EdgeHash is updated.
 *
 * A finite edge is complete after two distinct endpoint occurrences. An
 * unbounded edge contributes its persisted infinite-edge record as the second
 * occurrence. EdgeHashTable::all_edges_complete() additionally rejects a third
 * occurrence.
 *
 * This is a topological completeness check under the same full-dimensional
 * Voronoi assumptions as the construction algorithm. It cannot manufacture an
 * edge that is absent together with both of its endpoint/incidence records.
 */
template <class Mesh>
[[nodiscard]] MeshCompletenessReport<Mesh> verify_mesh_complete(
    Mesh& mesh,
    typename Mesh::NodeScalar maximum_variance =
        typename Mesh::NodeScalar{1e-20},
    bool print_errors = false,
    std::ostream& output = std::cerr) {

    using Index = typename Mesh::Index;
    using Address = typename Mesh::Address;
    using Sigma = typename Mesh::Sigma;
    using ExtendedNodes = std::remove_reference_t<
        decltype(std::declval<detail::MeshValidationContext<Mesh>&>()
                     .extended_nodes())>;
    using Iterator = EdgeIterator<ExtendedNodes, detail::EmptyLock>;
    using GlobalEdgeHash = detail::EdgeHashTable<detail::EmptyLock>;

    MeshCompletenessReport<Mesh> report;
    report.consistency = verify_mesh(
        mesh,
        maximum_variance,
        print_errors,
        output);

    detail::MeshValidationContext<Mesh> context(mesh);
    Iterator iterator(context.extended_nodes());

    // One persistent vertex can be visible in several cells. In degenerate
    // geometry those cell-local traversals may expose the same full supporting
    // edge through different minimal edges. Keep only one endpoint occurrence
    // per (vertex address, full edge).
    std::unordered_map<Address, std::vector<Sigma>> edges_by_vertex;

    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        context.activate_cell(cell);

        for (const auto& vertex : mesh.vertices(cell)) {
            iterator.reset(
                vertex.sigma,
                vertex.position,
                cell,
                typename Iterator::OnQueueEdges{});

            auto& full_edges = edges_by_vertex[vertex.address];

            while (const auto edge = iterator.next()) {
                Sigma full_edge(
                    edge->full_indices().begin(),
                    edge->full_indices().end());
                std::sort(full_edge.begin(), full_edge.end());

                const auto duplicate = std::find(
                    full_edges.begin(),
                    full_edges.end(),
                    full_edge);
                if (duplicate != full_edges.end()) {
                    ++report.duplicate_local_edge_representations;
                    continue;
                }

                full_edges.push_back(std::move(full_edge));
            }
        }
    }

    GlobalEdgeHash edge_hash(256);
    for (const auto& entry : edges_by_vertex) {
        for (const Sigma& full_edge : entry.second) {
            (void)edge_hash.pushedge(full_edge, std::int64_t{0}, true);
            ++report.unique_finite_edge_endpoints;
        }
    }

    for (const auto& infinite_edge : mesh.infinite_edges()) {
        (void)edge_hash.pushedge(
            infinite_edge.sigma,
            GlobalEdgeHash::infinite_cell,
            true);
        ++report.infinite_edges;
    }

    report.all_edges_have_two_occurrences =
        edge_hash.all_edges_complete();
    return report;
}

/**
 * @brief Compare nodes and primary Voronoi vertices of two compatible meshes.
 *
 * Membership is tested in both directions. Matching vertex signatures are
 * compared geometrically only once. The position tolerance is exactly the
 * rule requested by HighVoronoi's diagnostic workflow:
 *
 *     sqrt(max(vertex_variance(first), vertex_variance(second))).
 */
template <class FirstMesh, class SecondMesh>
[[nodiscard]] MeshComparisonReport<FirstMesh> compare_meshes(
    FirstMesh& first,
    SecondMesh& second,
    typename FirstMesh::NodeScalar node_tolerance =
        typename FirstMesh::NodeScalar{0},
    bool print_errors = false,
    std::ostream& output = std::cerr) {

    static_assert(
        std::is_same_v<typename FirstMesh::Index, typename SecondMesh::Index>,
        "compare_meshes requires equal Index types.");
    static_assert(
        std::is_same_v<
            typename FirstMesh::NodeScalar,
            typename SecondMesh::NodeScalar>,
        "compare_meshes requires equal node scalar types.");
    static_assert(
        std::is_same_v<
            typename FirstMesh::VertexScalar,
            typename SecondMesh::VertexScalar>,
        "compare_meshes requires equal vertex scalar types.");

    using Mesh = FirstMesh;
    using Index = typename Mesh::Index;
    using Scalar = typename Mesh::NodeScalar;
    using Error = MeshComparisonError<Mesh>;
    using VertexRecord = typename Mesh::VertexRecord;

    MeshComparisonReport<Mesh> report;

    auto add_error = [&](Error error) {
        if (print_errors) {
            detail::print_comparison_error(output, error);
        }
        report.errors.push_back(std::move(error));
    };

    if (first.dimension() != second.dimension()) {
        Error error;
        error.kind = MeshComparisonErrorKind::DimensionMismatch;
        add_error(std::move(error));
        return report;
    }

    if (first.size() != second.size()) {
        Error error;
        error.kind = MeshComparisonErrorKind::NodeCountMismatch;
        add_error(std::move(error));
        return report;
    }

    const std::size_t dimension =
        static_cast<std::size_t>(first.dimension());

    auto first_node = detail::make_validation_point<
        typename detail::ValidationRayCaster<FirstMesh>::Point>(dimension);
    auto second_node = detail::make_validation_point<
        typename detail::ValidationRayCaster<SecondMesh>::Point>(dimension);

    for (Index node = Index{0}; node < first.size(); ++node) {
        first.nodes().copy_node(node, first_node.data());
        second.nodes().copy_node(node, second_node.data());
        ++report.checked_nodes;

        const Scalar distance = (first_node - second_node).norm();
        if (distance > node_tolerance) {
            Error error;
            error.kind = MeshComparisonErrorKind::NodePositionMismatch;
            error.cell = node;
            error.distance = distance;
            error.tolerance = node_tolerance;
            add_error(std::move(error));
        }
    }

    detail::MeshValidationContext<FirstMesh> first_context(first);
    detail::MeshValidationContext<SecondMesh> second_context(second);

    auto first_position = detail::make_validation_point<
        typename detail::ValidationRayCaster<FirstMesh>::Point>(dimension);
    auto second_position = detail::make_validation_point<
        typename detail::ValidationRayCaster<SecondMesh>::Point>(dimension);

    for (Index cell = Index{0}; cell < first.size(); ++cell) {
        first_context.activate_cell(cell);
        second_context.activate_cell(cell);

        for (const auto& first_vertex : first.primary_vertices(cell)) {
            ++report.checked_primary_vertices;

            VertexRecord second_storage;
            second_storage.position = second.make_vertex_point();
            const auto* second_vertex = detail::find_primary_vertex(
                second,
                cell,
                first_vertex.sigma,
                second_storage);

            if (second_vertex == nullptr) {
                Error error;
                error.kind = MeshComparisonErrorKind::MissingInSecond;
                error.cell = cell;
                error.sigma = first_vertex.sigma;
                error.first_position = first_vertex.position;
                error.second_position = first.make_vertex_point();
                add_error(std::move(error));
                continue;
            }

            first_context.load_position(first_vertex.position);
            second_context.load_position(second_vertex->position);
            const Scalar first_variance =
                first_context.variance(first_vertex.sigma);
            const Scalar second_variance =
                second_context.variance(second_vertex->sigma);

            detail::copy_validation_point(
                first_vertex.position,
                first_position);
            detail::copy_validation_point(
                second_vertex->position,
                second_position);

            // vertex_variance() is scale-free: it measures the relative
            // variance of squared generator distances. Convert its square root
            // back to a length with the local Voronoi radius. A small
            // floating-point floor is additionally required because two valid
            // constructions may solve the same vertex in a different order and
            // therefore differ by a few ulps even when both variances round to
            // zero.
            first.nodes().copy_node(first_vertex.sigma.front(), first_node.data());
            second.nodes().copy_node(second_vertex->sigma.front(), second_node.data());

            const Scalar first_radius = (first_position - first_node).norm();
            const Scalar second_radius = (second_position - second_node).norm();
            const Scalar radius = std::max(first_radius, second_radius);

            using std::sqrt;
            const Scalar variance_tolerance =
                radius * sqrt(std::max(first_variance, second_variance));

            const Scalar coordinate_scale = std::max({
                Scalar{1},
                first_position.norm(),
                second_position.norm(),
                first_node.norm(),
                second_node.norm()});
            const Scalar roundoff_tolerance =
                Scalar{128} * std::numeric_limits<Scalar>::epsilon() *
                coordinate_scale;

            const Scalar tolerance =
                std::max(variance_tolerance, roundoff_tolerance);
            const Scalar distance =
                (first_position - second_position).norm();

            ++report.matched_primary_vertices;
            report.vertex_distance_sum += distance;
            report.vertex_distance_max =
                std::max(report.vertex_distance_max, distance);

            if (!(distance <= tolerance)) {
                Error error;
                error.kind = MeshComparisonErrorKind::VertexPositionMismatch;
                error.cell = cell;
                error.sigma = first_vertex.sigma;
                error.first_position = first_vertex.position;
                error.second_position = second_vertex->position;
                error.distance = distance;
                error.tolerance = tolerance;
                add_error(std::move(error));
            }
        }

        for (const auto& second_vertex : second.primary_vertices(cell)) {
            VertexRecord first_storage;
            first_storage.position = first.make_vertex_point();
            const auto* first_vertex = detail::find_primary_vertex(
                first,
                cell,
                second_vertex.sigma,
                first_storage);

            if (first_vertex == nullptr) {
                Error error;
                error.kind = MeshComparisonErrorKind::MissingInFirst;
                error.cell = cell;
                error.sigma = second_vertex.sigma;
                error.first_position = first.make_vertex_point();
                error.second_position = second_vertex.position;
                add_error(std::move(error));
            }
        }
    }

    return report;
}

} // namespace highvoronoi
