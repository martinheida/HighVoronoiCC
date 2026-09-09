
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/high_voronoi/compute_high_voronoi.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/high_voronoi_compute_mesh.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/mesh/validation/mesh_cell_validation.hpp>
#include <highvoronoi/mesh/visible_first_mesh.hpp>
#include <highvoronoi/core/detail/atomic_bit_vector.hpp>
#include <highvoronoi/search/detail/nanoflann.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <limits>

namespace {

// =============================================================================
// All template-heavy declarations are kept here.
// =============================================================================

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;
inline constexpr Index VisibleNodeCount = Index{30};

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, DatabaseParameters, Dimension>;

using HighMesh = highvoronoi::HighVoronoiMesh<
    Scalar,
    Dimension,
    Database>;
using ReferenceMesh = highvoronoi::VoronoiMesh<
    Scalar,
    Dimension,
    Database>;
using ReferenceNodes = ReferenceMesh::InternalNodes;

using Point = HighMesh::NodePoint;
using VertexPoint = HighMesh::VertexPoint;
using Boundary = HighMesh::BoundaryType;

using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::InRangeRaycast,
    Scalar>;

using HighCompute = highvoronoi::ComputeHighVoronoi<
    HighMesh,
    highvoronoi::geometry::KDSearch,
    RayParameters,
    highvoronoi::SingleThread,
    highvoronoi::SingleThread,
    DatabaseParameters,
    EdgeParameters>;

inline constexpr std::size_t DatabaseUnits = 65536;
inline constexpr std::size_t HashCapacity = 65536;
inline constexpr Scalar RayVarianceTolerance = Scalar{9e-14};
inline constexpr Scalar VerificationTolerance = Scalar{1e-12};
inline constexpr Scalar VertexMatchTolerance = Scalar{2e-9};
inline constexpr Scalar MinimumNodeDistance = Scalar{0.055};

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    ++performed_checks;
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

void section(std::string_view title) {
    std::cout << "\n============================================================\n"
              << title << '\n'
              << "============================================================\n";
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

DatabaseParameters database_parameters() {
    return DatabaseParameters{highvoronoi::DirectHash{HashCapacity}};
}

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        DatabaseUnits,
        database_parameters());
}

Boundary periodic_unit_cube() {
    // Axes are zero-based in C++: x and y periodic, z Dirichlet.
    return Boundary::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{Index{0}, Index{1}});
}

Boundary explicit_reference_box() {
    // A 3x3 tiling in the two periodic directions surrounds the central tile.
    // The outer x/y faces are ordinary Dirichlet faces but are too far away to
    // affect any Voronoi cell belonging to one of the 30 central generators.
    return Boundary::cuboid(
        point(3.0, 3.0, 1.0),
        point(-1.0, -1.0, 0.0),
        std::vector<Index>{});
}

bool far_enough(const Point& candidate, const std::vector<Point>& points) {
    for (const Point& existing : points) {
        if ((candidate - existing).norm() < MinimumNodeDistance) {
            return false;
        }
    }
    return true;
}

std::vector<Point> make_visible_points() {
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(VisibleNodeCount));

    // Fixed seed: reproducible, but still an unstructured general-position cloud.
    // We use almost the complete unit cube so the periodic boundaries are
    // genuinely exercised rather than merely present in the metadata.
    std::mt19937_64 random(0x504552494f444943ULL); // "PERIODIC"
    std::uniform_real_distribution<Scalar> coordinate(Scalar{0.02}, Scalar{0.98});

    while (points.size() < static_cast<std::size_t>(VisibleNodeCount)) {
        Point candidate = point(
            coordinate(random),
            coordinate(random),
            coordinate(random));
        if (far_enough(candidate, points)) {
            points.push_back(candidate);
        }
    }

    return points;
}

std::unique_ptr<HighMesh> make_high_mesh(const std::vector<Point>& points) {
    auto mesh = std::make_unique<HighMesh>(
        Index{Dimension},
        periodic_unit_cube(),
        std::in_place,
        DatabaseUnits,
        database_parameters());

    for (const Point& value : points) {
        (void)mesh->append_visible_node(value);
    }
    return mesh;
}

ReferenceNodes make_reference_nodes(const std::vector<Point>& visible_points) {
    // Two periodic axes require all combinations of shifts -1, 0, +1.
    // Central block first so reference cell i corresponds directly to visible i.
    const std::vector<std::pair<int, int>> tiles{
        { 0,  0},
        {-1,  0}, { 1,  0},
        { 0, -1}, { 0,  1},
        {-1, -1}, {-1,  1}, { 1, -1}, { 1,  1}
    };

    const Index total_nodes = static_cast<Index>(
        tiles.size() * visible_points.size());
    ReferenceNodes nodes(total_nodes);

    Index output = Index{0};
    for (const auto [shift_x, shift_y] : tiles) {
        for (const Point& source : visible_points) {
            Point translated = source;
            translated[0] += static_cast<Scalar>(shift_x);
            translated[1] += static_cast<Scalar>(shift_y);
            nodes.set(output++, translated);
        }
    }

    return nodes;
}

std::unique_ptr<ReferenceMesh> make_reference_mesh(
    const std::vector<Point>& visible_points) {
    return std::make_unique<ReferenceMesh>(
        make_reference_nodes(visible_points),
        explicit_reference_box(),
        make_database());
}

RayParameters ray_parameters() {
    RayParameters parameters;
    parameters.variance_tolerance = RayVarianceTolerance;
    return parameters;
}

void compute_reference_mesh(ReferenceMesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});
    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters());

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        ReferenceMesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        DatabaseParameters,
        EdgeParameters>;

    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,
        database_parameters(),
        EdgeParameters{highvoronoi::DirectHash{HashCapacity}});
    compute.compute();
}


using VisibleMesh = highvoronoi::VisibleFirstMesh<HighMesh>;

std::vector<Point> collect_visible_points(VisibleMesh& mesh) {
    std::vector<Point> result;
    result.reserve(static_cast<std::size_t>(mesh.visible_end()));
    for (Index node = Index{0}; node < mesh.visible_end(); ++node) {
        result.push_back(mesh.nodes().node(node));
    }
    return result;
}

struct VertexCloud {
    std::vector<VertexPoint> points;

    [[nodiscard]] std::size_t kdtree_get_point_count() const noexcept {
        return points.size();
    }

    [[nodiscard]] Scalar kdtree_get_pt(
        std::size_t index,
        std::size_t coordinate_index) const {
        return points.at(index)[static_cast<Eigen::Index>(coordinate_index)];
    }

    template <class BoundingBox>
    [[nodiscard]] bool kdtree_get_bbox(BoundingBox&) const noexcept {
        return false;
    }
};

using VertexDistance = nanoflann::L2_Simple_Adaptor<
    Scalar,
    VertexCloud,
    Scalar,
    std::size_t>;
using VertexKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    VertexDistance,
    VertexCloud,
    Dimension,
    std::size_t>;

struct VertexTree {
    explicit VertexTree(std::vector<VertexPoint> points)
        : cloud{std::move(points)},
          tree(
              Dimension,
              cloud,
              nanoflann::KDTreeSingleIndexAdaptorParams{10}) {
        tree.buildIndex();
    }

    VertexCloud cloud;
    VertexKDTree tree;
};

std::vector<VertexPoint> collect_primary_vertices(
    ReferenceMesh& mesh,
    Index cell_end) {
    std::vector<VertexPoint> result;
    for (Index cell = Index{0}; cell < cell_end; ++cell) {
        for (const auto& vertex : mesh.primary_vertices(cell)) {
            result.push_back(vertex.position);
        }
    }
    return result;
}

std::vector<VertexPoint> collect_visible_vertex_occurrences(
    VisibleMesh& mesh) {
    std::vector<VertexPoint> result;

    // Important: do NOT deduplicate by database address here. A geometric
    // vertex belongs to several visible cells and therefore legitimately
    // occurs several times while iterating the visible cells. The KDTree
    // comparison below is a coverage test, not a bijection of occurrences.
    for (Index cell = Index{0}; cell < mesh.visible_end(); ++cell) {
        for (const auto& vertex : mesh.vertices(cell)) {
            result.push_back(vertex.position);
        }
    }
    return result;
}

bool inside_visible_domain(const VertexPoint& position) {
    constexpr Scalar tolerance = Scalar{2e-12};
    for (Eigen::Index coordinate_index = 0;
         coordinate_index < position.size();
         ++coordinate_index) {
        if (position[coordinate_index] < -tolerance ||
            position[coordinate_index] > Scalar{1} + tolerance) {
            return false;
        }
    }
    return true;
}

/**
 * Canonical representative of a periodic vertex.
 *
 * x and y are periodic and are reduced modulo one. z is non-periodic and is
 * left untouched, so z=0 and z=1 boundary vertices remain distinct.
 *
 * Periodic seam points are represented by 0 rather than 1. Genuine Voronoi
 * vertices lie on such a seam only in nongeneric configurations; this merely
 * removes a representational ambiguity.
 */
VertexPoint canonical_periodic_position(VertexPoint position) {
    constexpr Scalar tolerance = Scalar{2e-12};

    for (Eigen::Index coordinate_index = 0;
         coordinate_index < Eigen::Index{2};
         ++coordinate_index) {
        Scalar value = position[coordinate_index];
        value -= std::floor(value);

        if (std::abs(value) <= tolerance ||
            std::abs(value - Scalar{1}) <= tolerance) {
            value = Scalar{0};
        }

        position[coordinate_index] = value;
    }

    return position;
}

std::vector<VertexPoint> canonicalize_periodic_positions(
    const std::vector<VertexPoint>& vertices) {
    std::vector<VertexPoint> result;
    result.reserve(vertices.size());

    for (const auto& position : vertices) {
        result.push_back(canonical_periodic_position(position));
    }

    return result;
}

std::vector<VertexPoint> deduplicate_positions(
    const std::vector<VertexPoint>& vertices,
    Scalar tolerance) {

    std::vector<VertexPoint> result;
    const Scalar tolerance_squared = tolerance * tolerance;

    for (const auto& position : vertices) {
        bool duplicate = false;
        for (const auto& existing : result) {
            if ((position - existing).squaredNorm() <= tolerance_squared) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            result.push_back(position);
        }
    }

    return result;
}

std::vector<VertexPoint> periodic_reference_classes(
    const std::vector<VertexPoint>& reference_vertices) {

    return deduplicate_positions(
        canonicalize_periodic_positions(reference_vertices),
        VertexMatchTolerance);
}

std::vector<VertexPoint> collect_public_high_vertex_occurrences(HighMesh& mesh) {
    std::vector<VertexPoint> result;

    // Same coverage semantics as for VisibleFirstMesh: the same geometric
    // vertex may be exposed by several visible cells and may therefore hit
    // the same KDTree entry repeatedly.
    for (Index cell = Index{0}; cell < mesh.visible_public_count(); ++cell) {
        for (const auto& vertex : mesh.vertices(cell)) {
            result.push_back(vertex.position);
        }
    }
    return result;
}

struct KDMatchReport {
    std::size_t reference_count = 0;
    std::size_t query_count = 0;
    std::size_t matched_occurrences = 0;
    std::size_t repeated_hits = 0;
    std::size_t missing = 0;
    std::size_t uncovered_reference = 0;
    Scalar maximum_distance = Scalar{0};
    std::vector<std::size_t> uncovered_indices;

    [[nodiscard]] bool success() const noexcept {
        return missing == 0 && uncovered_reference == 0;
    }
};

KDMatchReport match_against_tree(
    const std::vector<VertexPoint>& reference,
    const std::vector<VertexPoint>& query,
    Scalar tolerance,
    std::string_view label) {

    KDMatchReport report;
    report.reference_count = reference.size();
    report.query_count = query.size();

    if (reference.empty()) {
        report.missing = query.size();
        return report;
    }

    VertexTree tree(reference);
    highvoronoi::detail::BitVector found(reference.size(), false);
    const Scalar tolerance_squared = tolerance * tolerance;

    for (std::size_t query_index = 0; query_index < query.size(); ++query_index) {
        std::size_t nearest_index = 0;
        Scalar squared_distance = (std::numeric_limits<Scalar>::max)();
        const std::size_t count = tree.tree.knnSearch(
            query[query_index].data(),
            std::size_t{1},
            &nearest_index,
            &squared_distance);

        if (count != 1 || squared_distance > tolerance_squared) {
            ++report.missing;
            std::cerr << "    [" << label << "] no KDTree pendant for query vertex "
                      << query_index << " position=("
                      << std::setprecision(16)
                      << query[query_index][0] << ','
                      << query[query_index][1] << ','
                      << query[query_index][2] << ") nearest_distance="
                      << (count == 1 ? std::sqrt(squared_distance)
                                     : (std::numeric_limits<Scalar>::infinity)())
                      << '\n';
            continue;
        }

        const Scalar distance = std::sqrt(squared_distance);
        report.maximum_distance = std::max(report.maximum_distance, distance);

        if (found.test(nearest_index)) {
            ++report.repeated_hits;
        } else {
            found.set(nearest_index);
        }
        ++report.matched_occurrences;
    }

    for (std::size_t index = 0; index < found.size(); ++index) {
        if (!found.test(index)) {
            ++report.uncovered_reference;
            report.uncovered_indices.push_back(index);
            const auto& position = reference[index];
            std::cerr << "    [" << label << "] reference KDTree vertex " << index
                      << " was never found; position=("
                      << std::setprecision(16)
                      << position[0] << ',' << position[1] << ',' << position[2]
                      << ")\n";
        }
    }

    return report;
}


void print_position(const VertexPoint& position) {
    std::cout << '(' << std::setprecision(16)
              << position[0] << ','
              << position[1] << ','
              << position[2] << ')';
}

template <class SigmaLike>
void print_sigma(const SigmaLike& sigma) {
    std::cout << '{';
    bool first = true;
    for (const auto index : sigma) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << index;
    }
    std::cout << '}';
}

VertexPoint as_vertex_point(const Point& source) {
    VertexPoint result;
    for (Eigen::Index coordinate = 0; coordinate < Dimension; ++coordinate) {
        result[coordinate] = source[coordinate];
    }
    return result;
}

std::vector<VertexPoint> visible_node_positions(VisibleMesh& mesh) {
    std::vector<VertexPoint> result;
    result.reserve(static_cast<std::size_t>(mesh.visible_end()));
    for (Index node = Index{0}; node < mesh.visible_end(); ++node) {
        result.push_back(as_vertex_point(mesh.nodes().node(node)));
    }
    return result;
}

template <class Mesh>
void inspect_target_in_cell(
    Mesh& mesh,
    Index cell,
    const VertexPoint& target,
    bool compare_periodically,
    std::string_view label) {

    const VertexPoint comparison_target =
        compare_periodically ? canonical_periodic_position(target) : target;

    bool found = false;
    Scalar best_distance = (std::numeric_limits<Scalar>::max)();
    std::optional<VertexPoint> best_position;
    std::vector<Index> best_sigma;
    std::size_t vertex_occurrences = 0;

    for (const auto& vertex : mesh.vertices(cell)) {
        ++vertex_occurrences;
        const VertexPoint comparison_position =
            compare_periodically
                ? canonical_periodic_position(vertex.position)
                : vertex.position;
        const Scalar distance =
            (comparison_position - comparison_target).norm();

        if (distance < best_distance) {
            best_distance = distance;
            best_position = vertex.position;
            best_sigma.assign(vertex.sigma.begin(), vertex.sigma.end());
        }

        if (distance <= VertexMatchTolerance) {
            found = true;
        }
    }

    std::cout << "        " << label
              << ": vertex_occurrences=" << vertex_occurrences
              << " target_stored=" << (found ? "YES" : "NO");

    if (best_position) {
        std::cout << " nearest_stored_distance=" << std::setprecision(16)
                  << best_distance << " nearest_stored_position=";
        print_position(*best_position);
        std::cout << " nearest_stored_sigma=";
        print_sigma(best_sigma);
    }
    std::cout << '\n';
}

void diagnose_missing_reference_vertices(
    VisibleMesh& visible_first,
    HighMesh& high_mesh,
    const std::vector<VertexPoint>& reference_primary,
    const KDMatchReport& exact_lift_report) {

    if (exact_lift_report.uncovered_indices.empty()) {
        std::cout << "    no uncovered exact-lift reference vertices to diagnose\n";
        return;
    }

    const auto nodes = visible_node_positions(visible_first);
    VertexTree node_tree(nodes);

    std::cout << "    diagnostic KDTree: " << nodes.size()
              << " visible generators\n";

    for (std::size_t missing_number = 0;
         missing_number < exact_lift_report.uncovered_indices.size();
         ++missing_number) {

        const std::size_t reference_index =
            exact_lift_report.uncovered_indices[missing_number];
        const VertexPoint& target = reference_primary.at(reference_index);

        std::size_t nearest_index = 0;
        Scalar nearest_squared = (std::numeric_limits<Scalar>::max)();
        const std::size_t nearest_count = node_tree.tree.knnSearch(
            target.data(),
            std::size_t{1},
            &nearest_index,
            &nearest_squared);

        std::cout << "\n    missing reference vertex #" << (missing_number + 1)
                  << " KDTree-index=" << reference_index << " position=";
        print_position(target);
        std::cout << '\n';

        if (nearest_count != 1) {
            std::cout << "      nearest-node search FAILED\n";
            continue;
        }

        const Scalar nearest_distance = std::sqrt(nearest_squared);
        const Scalar radius_slack = std::max(
            Scalar{1e-10},
            nearest_distance * Scalar{1e-9});
        const Scalar expanded_radius = nearest_distance + radius_slack;

        std::cout << "      nearest visible node: " << nearest_index
                  << " distance=" << std::setprecision(16)
                  << nearest_distance << " position=";
        print_position(nodes[nearest_index]);
        std::cout << '\n'
                  << "      expanded radius:      " << expanded_radius
                  << " (slack=" << radius_slack << ")\n";

        std::vector<nanoflann::ResultItem<std::size_t, Scalar>> neighbors;
        neighbors.reserve(8);
        nanoflann::SearchParameters search_parameters;
        search_parameters.sorted = true;
        const std::size_t neighbor_count = node_tree.tree.radiusSearch(
            target.data(),
            expanded_radius * expanded_radius,
            neighbors,
            search_parameters);

        std::cout << "      nodes within expanded circumradius: "
                  << neighbor_count << '\n';

        if (std::abs(target[2]) <= Scalar{2e-12}) {
            std::cout << "      boundary support: z=0 non-periodic plane\n";
        }
        if (std::abs(target[2] - Scalar{1}) <= Scalar{2e-12}) {
            std::cout << "      boundary support: z=1 non-periodic plane\n";
        }

        for (const auto& neighbor : neighbors) {
            const Index visible_node = static_cast<Index>(neighbor.first);
            const Scalar distance = std::sqrt(neighbor.second);

            std::cout << "\n      visible node " << visible_node
                      << " distance=" << std::setprecision(16) << distance
                      << " delta_to_nearest=" << (distance - nearest_distance)
                      << " position=";
            print_position(nodes[neighbor.first]);
            std::cout << '\n';

            // VisibleFirstMesh exposes the internal, unprojected address lists
            // of the canonical visible node. Compare modulo periodic x/y
            // translations as well, because the stored lift is not prescribed.
            inspect_target_in_cell(
                visible_first,
                visible_node,
                target,
                false,
                "VisibleFirst exact lift");
            inspect_target_in_cell(
                visible_first,
                visible_node,
                target,
                true,
                "VisibleFirst periodic class");

            // Same visible insertion order on the public HighVoronoi side.
            inspect_target_in_cell(
                high_mesh,
                visible_node,
                target,
                true,
                "HighVoronoi public projected class");
        }
    }
}

void print_match_report(std::string_view label, const KDMatchReport& report) {
    std::cout << "    " << label << ":\n"
              << "      reference KDTree vertices: " << report.reference_count << '\n'
              << "      query vertices:            " << report.query_count << '\n'
              << "      matched query occurrences: " << report.matched_occurrences << '\n'
              << "      repeated KDTree hits:       " << report.repeated_hits << '\n'
              << "      missing queries:            " << report.missing << '\n'
              << "      uncovered reference:        " << report.uncovered_reference << '\n'
              << "      maximum position error:    " << std::setprecision(16)
              << report.maximum_distance << '\n';
}

} // namespace

int main() {
    section("HighVoronoi 3D periodic reference and representation test");
    std::cout << "Visible domain: [0,1]^3\n"
              << "Periodic axes:  x (0), y (1)\n"
              << "Non-periodic:   z (2)\n"
              << "Visible nodes:  30\n"
              << "Reference tiling: 3 x 3 x 1 = 9 copies = 270 nodes\n";

    const std::vector<Point> visible_points = make_visible_points();

    // ---------------------------------------------------------------------
    // 1. Compute HighVoronoi and validate the complete visible-cell prefix.
    // ---------------------------------------------------------------------
    section("1. HighVoronoi visible cells: consistency and completeness");

    auto high_mesh = make_high_mesh(visible_points);
    check(high_mesh->size() == VisibleNodeCount,
          "HighVoronoi initially exposes exactly the 30 visible nodes");

    HighCompute high_compute(
        *high_mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters(),
        EdgeParameters{highvoronoi::DirectHash{HashCapacity}});

    const auto high_report = high_compute.compute();
    std::cout << "    compute rounds:        " << high_report.rounds << '\n'
              << "    visible nodes:         " << high_mesh->visible_public_count() << '\n'
              << "    active internal nodes: " << high_mesh->internal_node_count() << '\n';

    VisibleMesh visible_first(*high_mesh);
    check(visible_first.visible_end() == VisibleNodeCount,
          "VisibleFirstMesh exposes the 30 visible nodes as its exact prefix");

    bool visible_prefix_order_ok = true;
    for (Index visible = Index{0}; visible < visible_first.visible_end(); ++visible) {
        visible_prefix_order_ok &=
            visible_first.global_internal_node(visible) ==
            high_mesh->visible_public_to_internal(visible);
    }
    check(visible_prefix_order_ok,
          "VisibleFirstMesh preserves visible insertion order in its prefix");

    const auto visible_complete = highvoronoi::verify_mesh_complete_cells(
        visible_first,
        Index{0},
        visible_first.visible_end(),
        VerificationTolerance,
        true,
        std::cerr,
        HashCapacity);

    std::cout << "    visible checked vertex occurrences: "
              << visible_complete.consistency.checked_vertex_occurrences << '\n'
              << "    visible consistency errors:         "
              << visible_complete.consistency.error_count() << '\n'
              << "    visible finite edge endpoints:      "
              << visible_complete.unique_finite_edge_endpoints << '\n'
              << "    visible edge closure:               "
              << (visible_complete.all_edges_have_two_occurrences ? "YES" : "NO")
              << '\n';

    check(visible_complete.consistency.valid(),
          "all vertices stored in visible cells are geometrically consistent");
    check(visible_complete.complete(),
          "all visible cells are topologically complete");

    // ---------------------------------------------------------------------
    // 2. All active internal vertices must at least be individually valid.
    // Invisible periodic reference cells need not be complete.
    // ---------------------------------------------------------------------
    section("2. HighVoronoi internally consistent");

    const auto internal_consistency = highvoronoi::verify_mesh(
        visible_first,
        VerificationTolerance,
        true);

    std::cout << "    all-active checked vertex occurrences: "
              << internal_consistency.checked_vertex_occurrences << '\n'
              << "    all-active consistency errors:         "
              << internal_consistency.error_count() << '\n';

    check(internal_consistency.valid(),
          "every stored vertex in the complete active internal mesh is consistent");

    // ---------------------------------------------------------------------
    // Build the independent explicit periodic reference once.
    // ---------------------------------------------------------------------
    const auto reference_visible_points = collect_visible_points(visible_first);
    auto reference_mesh = make_reference_mesh(reference_visible_points);
    compute_reference_mesh(*reference_mesh);

    const auto reference_consistency = highvoronoi::verify_mesh(
        *reference_mesh,
        VerificationTolerance,
        true);
    check(reference_consistency.valid(),
          "explicit 3x3 periodic reference VoronoiMesh is geometrically consistent");

    // ---------------------------------------------------------------------
    // 3. Internal periodic geometry.
    //
    // The explicit 3x3 mesh and HighVoronoi are allowed to choose different
    // lifts of the same torus vertex. Exact coordinates in the universal cover
    // are therefore diagnostic only. The invariant is the vertex set modulo
    // integer translations in the periodic x/y directions.
    // ---------------------------------------------------------------------
    section("3. VisibleFirstMesh against periodic-equivalence reference");

    const auto reference_primary = collect_primary_vertices(
        *reference_mesh,
        VisibleNodeCount);
    const auto high_internal_vertices = collect_visible_vertex_occurrences(
        visible_first);

    const auto exact_lift_match = match_against_tree(
        reference_primary,
        high_internal_vertices,
        VertexMatchTolerance,
        "VisibleFirstMesh/exact-lift-diagnostic");
    print_match_report("exact-lift diagnostic (not a pass criterion)",
                       exact_lift_match);

    section("3b. Diagnose uncovered reference vertices against the 30 visible generators");
    diagnose_missing_reference_vertices(
        visible_first,
        *high_mesh,
        reference_primary,
        exact_lift_match);

    const auto reference_periodic =
        periodic_reference_classes(reference_primary);
    const auto high_internal_periodic =
        canonicalize_periodic_positions(high_internal_vertices);

    const auto internal_match = match_against_tree(
        reference_periodic,
        high_internal_periodic,
        VertexMatchTolerance,
        "VisibleFirstMesh/periodic-equivalence");
    print_match_report("periodic-equivalence representation", internal_match);
    check(internal_match.success(),
          "VisibleFirstMesh visible-cell vertices cover every periodic reference vertex class");

    // ---------------------------------------------------------------------
    // 4. Public HighVoronoi representation.
    //
    // Public HighVoronoi explicitly translates stored vertices into the
    // external periodic domain. The correct reference is therefore NOT the
    // subset of raw reference lifts already lying in [0,1]^3. Every periodic
    // reference vertex must first be reduced modulo the x/y lattice.
    // ---------------------------------------------------------------------
    section("4. Public HighVoronoi representation inside visible domain");

    const auto public_high_vertices =
        collect_public_high_vertex_occurrences(*high_mesh);

    bool public_inside = true;
    for (const auto& position : public_high_vertices) {
        public_inside &= inside_visible_domain(position);
    }
    check(public_inside,
          "every public HighVoronoi vertex lies inside the visible domain");

    const auto public_match = match_against_tree(
        reference_periodic,
        public_high_vertices,
        VertexMatchTolerance,
        "HighVoronoi-public/projected-reference");
    print_match_report("public visible-domain representation", public_match);
    check(public_match.success(),
          "public HighVoronoi vertices cover the projected periodic reference set");

    section("Result");
    std::cout << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}


