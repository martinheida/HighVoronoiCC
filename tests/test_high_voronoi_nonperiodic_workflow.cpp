
#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/detail/hybrid_database.hpp>
#include <highvoronoi/geometry/compute_high_voronoi.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/cuboid_mesh_engine.hpp>
#include <highvoronoi/geometry/high_voronoi_mesh.hpp>
#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

// -----------------------------------------------------------------------------
// All template-heavy declarations live here. The actual tests below should read
// in terms of SourceMesh, HighMesh, CuboidEngine, Point, and Index only.
// -----------------------------------------------------------------------------

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using StoredDatabase = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters>;
using HybridDatabase = highvoronoi::HybridDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters>;

using SourceMesh = highvoronoi::VoronoiMesh<
    Scalar,
    Dimension,
    StoredDatabase>;
using SourceNodes = SourceMesh::InternalNodes;

using HighMesh = highvoronoi::HighVoronoiMesh<
    Scalar,
    Dimension,
    HybridDatabase>;
using HighDataView = highvoronoi::HighVoronoiDataMeshView<HighMesh>;
using HighCompute = highvoronoi::ComputeHighVoronoi<HighMesh>;

using CuboidEngine = highvoronoi::CuboidMeshEngine<
    Scalar,
    Scalar,
    Index,
    Dimension>;
using GridCounts = CuboidEngine::CountPoint;

using Point = HighMesh::NodePoint;
using Boundary = HighMesh::BoundaryType;
using Address = HighMesh::Address;

inline constexpr std::size_t StoredUnits = 65536;
inline constexpr std::size_t HashCapacity = 65536;
inline constexpr Scalar VerificationTolerance = Scalar{1e-14};

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

Boundary cube_boundary() {
    return Boundary::cuboid(
        point(2.0, 2.0, 2.0),
        point(-1.0, -1.0, -1.0),
        std::vector<Index>{}); // no periodic axes
}

DatabaseParameters database_parameters() {
    return DatabaseParameters{
        highvoronoi::DirectHash{HashCapacity}};
}

std::shared_ptr<StoredDatabase> make_source_database() {
    return std::make_shared<StoredDatabase>(
        StoredUnits,
        database_parameters());
}

std::unique_ptr<HighMesh> make_high_mesh() {
    return std::make_unique<HighMesh>(
        Index{Dimension},
        cube_boundary(),
        StoredUnits,
        database_parameters());
}

bool in_negative_octant(const Point& value) {
    return value[0] <= Scalar{0} &&
           value[1] <= Scalar{0} &&
           value[2] <= Scalar{0};
}

bool in_positive_octant(const Point& value) {
    return value[0] >= Scalar{0} &&
           value[1] >= Scalar{0} &&
           value[2] >= Scalar{0};
}

bool far_enough(
    const Point& candidate,
    const std::vector<Point>& points,
    Scalar minimum_distance = Scalar{0.12}) {
    for (const Point& existing : points) {
        if ((candidate - existing).norm() < minimum_distance) {
            return false;
        }
    }
    return true;
}

std::vector<Point> make_source_points() {
    std::vector<Point> points;
    points.reserve(10);

    // Required deterministic anchors: origin, one genuinely negative-octant
    // point, and one genuinely positive-octant point.
    points.push_back(point(0.0, 0.0, 0.0));
    points.push_back(point(-0.72, -0.58, -0.44));
    points.push_back(point(0.63, 0.71, 0.52));

    std::mt19937_64 random(0x4856534f55524345ULL);
    std::uniform_real_distribution<Scalar> coordinate(-0.82, 0.82);

    while (points.size() < 10) {
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

std::vector<Point> make_negative_high_points() {
    std::vector<Point> points;
    points.reserve(10);

    std::mt19937_64 random(0x48564e4547415449ULL);
    std::uniform_real_distribution<Scalar> coordinate(-0.92, -0.08);

    while (points.size() < 10) {
        Point candidate = point(
            coordinate(random),
            coordinate(random),
            coordinate(random));
        if (far_enough(candidate, points, Scalar{0.10})) {
            points.push_back(candidate);
        }
    }
    return points;
}

std::vector<Point> make_engine_high_points() {
    std::vector<Point> points;
    points.reserve(10);

    // Nine nodes are strictly in the negative octant.
    std::mt19937_64 random(0x4856454e47494e45ULL);
    std::uniform_real_distribution<Scalar> coordinate(-0.90, -0.10);
    while (points.size() < 9) {
        Point candidate = point(
            coordinate(random),
            coordinate(random),
            coordinate(random));
        if (far_enough(candidate, points, Scalar{0.10})) {
            points.push_back(candidate);
        }
    }

    // Exactly one pre-existing node lies in [0,1]^3. It is deliberately not a
    // Cartesian-grid node and should invalidate nearby analytic grid vertices.
    points.push_back(point(0.53, 0.47, 0.56));
    return points;
}

SourceNodes make_source_nodes(const std::vector<Point>& points) {
    SourceNodes nodes(static_cast<Index>(points.size()));
    for (Index i = Index{0}; i < static_cast<Index>(points.size()); ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
    }
    return nodes;
}

void compute_source_mesh(SourceMesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    highvoronoi::RaycastParameters<highvoronoi::ClassicRaycast, Scalar>
        ray_parameters;
    ray_parameters.variance_tolerance = Scalar{1e-12};
    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters);

    using RayCaster = decltype(raycaster);
    using EdgeParameters = highvoronoi::EdgeBufferParams<>;
    using Compute = highvoronoi::ComputeVoronoi<
        SourceMesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        DatabaseParameters,
        EdgeParameters>;

    const DatabaseParameters queue_parameters{
        highvoronoi::DirectHash{2048}};
    const EdgeParameters edge_parameters{
        highvoronoi::DirectHash{4096}};

    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,
        queue_parameters,
        edge_parameters);
    compute.compute();
}

std::unique_ptr<SourceMesh> make_computed_source_mesh() {
    const std::vector<Point> points = make_source_points();
    auto mesh = std::make_unique<SourceMesh>(
        make_source_nodes(points),
        cube_boundary(),
        make_source_database());
    compute_source_mesh(*mesh);
    return mesh;
}

void append_points(HighMesh& mesh, const std::vector<Point>& points) {
    for (const Point& value : points) {
        (void)mesh.append_visible_node(value);
    }
}

template <class MeshLike>
std::size_t unique_primary_vertex_count(const MeshLike& mesh) {
    std::unordered_set<typename MeshLike::Address> addresses;
    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        for (const auto& vertex : mesh.primary_vertices(cell)) {
            addresses.insert(vertex.address);
        }
    }
    return addresses.size();
}

void check_data_view_is_stable_order(HighMesh& mesh, std::string_view name) {
    auto view = mesh.data_mesh();

    Index previous = Index{0};
    bool first = true;
    bool correct = true;
    std::size_t active_count = 0;

    for (Index internal = Index{0};
         internal < mesh.internal_node_count();
         ++internal) {
        if (mesh.is_active_internal(internal)) {
            ++active_count;
        }
    }

    if (view.size() != static_cast<Index>(active_count)) {
        correct = false;
    }

    for (Index public_node = Index{0}; public_node < view.size(); ++public_node) {
        const Index internal = view.global_internal_node(public_node);
        if (!mesh.is_active_internal(internal)) {
            correct = false;
        }
        if (!first && internal <= previous) {
            correct = false;
        }
        previous = internal;
        first = false;
    }

    std::cout << "    data view '" << name << "': "
              << view.size() << " active nodes from "
              << mesh.internal_node_count() << " stable internal slots\n";
    check(correct,
          "data view contains every active node exactly in insertion order");
}

void print_consistency(
    std::string_view name,
    HighMesh& mesh) {
    auto view = mesh.data_mesh();
    const auto report = highvoronoi::verify_mesh(
        view,
        VerificationTolerance,
        true);

    std::cout << "    " << name << ": checked "
              << report.checked_vertex_occurrences
              << " vertex occurrences, errors = "
              << report.error_count() << '\n';
    check(report.valid(),
          "all currently stored vertices are geometrically consistent");
}

void print_completeness(
    std::string_view name,
    HighMesh& mesh) {
    auto view = mesh.data_mesh();
    const auto report = highvoronoi::verify_mesh_complete(
        view,
        VerificationTolerance,
        true);

    std::cout << "    " << name << ": finite edge endpoints = "
              << report.unique_finite_edge_endpoints
              << ", infinite edges = " << report.infinite_edges
              << ", edge closure = "
              << (report.all_edges_have_two_occurrences ? "yes" : "no")
              << '\n';

    check(report.consistency.valid(),
          "computed mesh is geometrically consistent");
    check(report.complete(),
          "computed bounded mesh is topologically complete");
}



Index flattened_grid_node(Index x, Index y, Index z) {
    constexpr Index side = Index{10};
    return static_cast<Index>(x + side * y + side * side * z);
}

} // namespace

int main() {
    section("HighVoronoi non-periodic 3D workflow test");
    std::cout << "External domain: [-1,+1]^3\n"
              << "All boundaries are non-periodic.\n";

    // =========================================================================
    // TEST 1: append an already-computed VoronoiMesh to a HighVoronoiMesh.
    // =========================================================================
    section("1. Import a computed VoronoiMesh");

    std::cout << "[source] create 10 nodes, including origin and nodes in both octants\n";
    auto source_mesh = make_computed_source_mesh();

    bool source_has_origin = false;
    bool source_has_negative = false;
    bool source_has_positive = false;
    for (Index node = Index{0}; node < source_mesh->size(); ++node) {
        Point value;
        source_mesh->nodes().copy_node(node, value.data());
        source_has_origin = source_has_origin || value.norm() == Scalar{0};
        source_has_negative = source_has_negative || in_negative_octant(value);
        source_has_positive = source_has_positive || in_positive_octant(value);
    }
    check(source_has_origin, "source VoronoiMesh contains the origin");
    check(source_has_negative, "source VoronoiMesh contains a node in [-1,0]^3");
    check(source_has_positive, "source VoronoiMesh contains a node in [0,1]^3");

    const auto source_complete = highvoronoi::verify_mesh_complete(
        *source_mesh,
        VerificationTolerance,
        true);
    check(source_complete.complete(),
          "source VoronoiMesh is complete before import");

    auto imported_mesh = make_high_mesh();
    const auto first_points = make_negative_high_points();
    append_points(*imported_mesh, first_points);
    check(imported_mesh->size() == Index{10},
          "HighVoronoiMesh initially exposes ten visible nodes");

    std::cout << "[import] append source nodes and only source vertices still valid\n"
              << "         against all 20 active nodes\n";
    const auto import_report = imported_mesh->append_mesh(*source_mesh);

    std::cout << "    appended nodes:            " << import_report.appended_nodes << '\n'
              << "    accepted source vertices:  " << import_report.accepted_vertices << '\n'
              << "    rejected invalid vertices: " << import_report.rejected_invalid_vertices << '\n'
              << "    duplicate vertices:        " << import_report.duplicate_vertices << '\n';

    check(import_report.appended_nodes == 10,
          "exactly ten source nodes were appended");
    check(imported_mesh->size() == Index{20},
          "HighVoronoiMesh now exposes twenty visible nodes");
    check_data_view_is_stable_order(*imported_mesh, "mesh-import state");
    print_consistency("mesh-import state", *imported_mesh);

    // =========================================================================
    // TEST 2: append an analytic 0.1-spaced Cartesian engine in [0,1]^3.
    // =========================================================================
    section("2. Import a CuboidMeshEngine");

    auto engine_mesh = make_high_mesh();
    const auto engine_background_points = make_engine_high_points();
    append_points(*engine_mesh, engine_background_points);

    std::size_t positive_before_engine = 0;
    for (Index node = Index{0}; node < engine_mesh->size(); ++node) {
        Point value;
        engine_mesh->nodes().copy_node(node, value.data());
        if (in_positive_octant(value)) {
            ++positive_before_engine;
        }
    }
    check(positive_before_engine == 1,
          "exactly one pre-existing HighVoronoi node lies in [0,1]^3");

    const Point grid_start = point(0.05, 0.05, 0.05);
    const Point grid_spacing = point(0.1, 0.1, 0.1);
    GridCounts grid_counts;
    grid_counts << Index{10}, Index{10}, Index{10};

    constexpr std::size_t perfect_grid_vertices = 9u * 9u * 9u;
    std::cout << "[engine] append 10^3 grid nodes at 0.05,...,0.95 and "
              << perfect_grid_vertices << " analytic finite vertices\n";

    CuboidEngine engine(grid_start, grid_spacing, grid_counts);
    const auto engine_report = engine_mesh->append_engine(std::move(engine));

    auto engine_data = engine_mesh->data_mesh();
    const std::size_t active_engine_vertices =
        unique_primary_vertex_count(engine_data);

    std::cout << "    appended grid nodes:        " << engine_report.appended_nodes << '\n'
              << "    perfect grid vertices:      " << perfect_grid_vertices << '\n'
              << "    accepted engine vertices:   " << engine_report.accepted_vertices << '\n'
              << "    rejected invalid vertices:  " << engine_report.rejected_invalid_vertices << '\n'
              << "    actual active vertices:     " << active_engine_vertices << '\n';

    check(engine_report.appended_nodes == Index{1000},
          "the engine contributed all 10^3 Cartesian nodes");
    check(engine_report.rejected_invalid_vertices > 0,
          "the pre-existing positive node invalidated some grid vertices");
    check(engine_report.accepted_vertices < perfect_grid_vertices,
          "fewer vertices survived than in the perfect isolated grid");
    check(active_engine_vertices == engine_report.accepted_vertices,
          "the data view exposes exactly the surviving engine vertices");
    check_data_view_is_stable_order(*engine_mesh, "engine-import state");
    print_consistency("engine-import state", *engine_mesh);

    // =========================================================================
    // TEST 3: mark a few visible nodes for deletion. The mesh is intentionally
    // not repaired yet; only node state and the stable-order data view are tested.
    // =========================================================================
    section("3. Mark nodes for deletion");

    std::cout << "[delete] remove three nodes from the mesh-import scenario\n";
    const auto removed_import = imported_mesh->erase_visible_nodes(
        std::vector<Index>{Index{1}, Index{7}, Index{15}});
    std::cout << "    removed internal nodes: "
              << removed_import.removed_internal.size() << '\n';
    check(removed_import.removed_internal.size() == 3,
          "three non-periodic nodes were marked inactive");
    check(imported_mesh->size() == Index{17},
          "seventeen visible nodes remain in the mesh-import scenario");
    check_data_view_is_stable_order(*imported_mesh, "mesh-import after delete");

    std::cout << "[delete] remove one original node and two interior grid nodes\n";
    const Index engine_node_offset = Index{10};
    const Index grid_center = static_cast<Index>(
        engine_node_offset + flattened_grid_node(5, 5, 5));
    const Index grid_upper = static_cast<Index>(
        engine_node_offset + flattened_grid_node(7, 7, 7));

    const auto removed_engine = engine_mesh->erase_visible_nodes(
        std::vector<Index>{Index{2}, grid_center, grid_upper});
    std::cout << "    removed internal nodes: "
              << removed_engine.removed_internal.size() << '\n';
    check(removed_engine.removed_internal.size() == 3,
          "three engine-scenario nodes were marked inactive");
    check_data_view_is_stable_order(*engine_mesh, "engine-import after delete");

    // =========================================================================
    // TEST 4: give both dirty/incomplete states to ComputeHighVoronoi. For a
    // bounded non-periodic cube the final result should be both consistent and
    // topologically complete; no persisted infinite edges are required.
    // =========================================================================
    section("4. Repair both HighVoronoi meshes with ComputeHighVoronoi");

    std::cout << "[compute A] repair mesh-import scenario\n";
    HighCompute compute_import(*imported_mesh);
    const auto compute_import_report = compute_import.compute();
    std::cout << "    rounds:                    " << compute_import_report.rounds << '\n'
              << "    deleted vertices:          " << compute_import_report.deleted_vertices << '\n'
              << "    invalidated old vertices:  " << compute_import_report.invalidated_old_vertices << '\n'
              << "    new-cell vertices:         " << compute_import_report.new_cell_vertices << '\n'
              << "    repair vertices:           " << compute_import_report.repair_vertices << '\n'
              << "    new reference nodes:       " << compute_import_report.new_reference_nodes << '\n'
              << "    final visible nodes:       " << compute_import_report.final_visible_nodes << '\n';
    check_data_view_is_stable_order(*imported_mesh, "mesh-import final");
    print_completeness("mesh-import final", *imported_mesh);

    std::cout << "\n[compute B] repair engine-import scenario\n";
    HighCompute compute_engine(*engine_mesh);
    const auto compute_engine_report = compute_engine.compute();
    std::cout << "    rounds:                    " << compute_engine_report.rounds << '\n'
              << "    deleted vertices:          " << compute_engine_report.deleted_vertices << '\n'
              << "    invalidated old vertices:  " << compute_engine_report.invalidated_old_vertices << '\n'
              << "    new-cell vertices:         " << compute_engine_report.new_cell_vertices << '\n'
              << "    repair vertices:           " << compute_engine_report.repair_vertices << '\n'
              << "    new reference nodes:       " << compute_engine_report.new_reference_nodes << '\n'
              << "    final visible nodes:       " << compute_engine_report.final_visible_nodes << '\n';
    check_data_view_is_stable_order(*engine_mesh, "engine-import final");
    print_completeness("engine-import final", *engine_mesh);

    section("Result");
    std::cout << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}


