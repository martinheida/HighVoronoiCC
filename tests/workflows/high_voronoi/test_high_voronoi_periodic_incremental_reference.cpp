


#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/high_voronoi/compute_high_voronoi.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/high_voronoi_compute_mesh.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// =============================================================================
// Template-heavy declarations stay here so the actual workflow remains readable.
// =============================================================================

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;
inline constexpr Index InitialVisibleNodeCount = Index{28};
inline constexpr Index AddedVisibleNodeCount = Index{3};
inline constexpr Index FinalVisibleNodeCount = Index{30};

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, DatabaseParameters, Dimension>;

using HighMesh = highvoronoi::HighVoronoiMesh<
    Scalar,
    Dimension,
    Database>;
using ComputeRoundState = highvoronoi::HighVoronoiComputeRoundState<HighMesh, highvoronoi::detail::BitVector>;
using ComputeMesh = highvoronoi::HighVoronoiComputeMesh<HighMesh, highvoronoi::detail::BitVector>;

using ReferenceMesh = highvoronoi::VoronoiMesh<
    Scalar,
    Dimension,
    Database>;
using ReferenceNodes = ReferenceMesh::InternalNodes;

using Point = HighMesh::NodePoint;
using VertexPoint = HighMesh::VertexPoint;
using Address = HighMesh::Address;
using Boundary = HighMesh::BoundaryType;
using NormalizedSignature = std::vector<std::uint64_t>;

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
inline constexpr std::uint64_t BoundaryTagBase = std::uint64_t{1} << 48U;

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

RayParameters ray_parameters() {
    RayParameters parameters;
    parameters.variance_tolerance = RayVarianceTolerance;
    return parameters;
}

Boundary periodic_unit_cube() {
    // C++ axes are zero-based: x and y periodic, z non-periodic.
    return Boundary::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{Index{0}, Index{1}});
}

Boundary explicit_reference_box() {
    // Full 3x3 tiling in the periodic x/y directions.
    return Boundary::cuboid(
        point(3.0, 3.0, 1.0),
        point(-1.0, -1.0, 0.0),
        std::vector<Index>{});
}

bool far_enough(
    const Point& candidate,
    const std::vector<Point>& points,
    Scalar minimum_distance = MinimumNodeDistance) {
    for (const Point& existing : points) {
        if ((candidate - existing).norm() < minimum_distance) {
            return false;
        }
    }
    return true;
}

std::vector<Point> refinement_points() {
    // Three mutually well-separated points close to periodic faces.
    // They are deliberately inserted only after the first HighVoronoi compute.
    return {
        point(0.008, 0.18, 0.37), // close to x = 0
        point(0.992, 0.73, 0.63), // close to x = 1
        point(0.43, 0.009, 0.84)  // close to y = 0
    };
}

std::vector<Point> make_initial_visible_points() {
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(InitialVisibleNodeCount));

    // Keep the random background away from the three future refinement points
    // so the incremental test never degenerates because two generators almost
    // coincide. The cloud still fills almost the entire unit cube.
    const std::vector<Point> future = refinement_points();

    std::mt19937_64 random(0x494e4352454d454eULL); // "INCREMEN"
    std::uniform_real_distribution<Scalar> coordinate(Scalar{0.025}, Scalar{0.975});

    while (points.size() < static_cast<std::size_t>(InitialVisibleNodeCount)) {
        Point candidate = point(
            coordinate(random),
            coordinate(random),
            coordinate(random));

        if (!far_enough(candidate, points)) {
            continue;
        }
        if (!far_enough(candidate, future, Scalar{0.085})) {
            continue;
        }
        points.push_back(candidate);
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

void append_visible_points(HighMesh& mesh, const std::vector<Point>& points) {
    for (const Point& value : points) {
        (void)mesh.append_visible_node(value);
    }
}

std::vector<Point> collect_visible_points(const HighMesh& mesh) {
    std::vector<Point> result;
    result.reserve(static_cast<std::size_t>(mesh.size()));

    for (Index public_node = Index{0}; public_node < mesh.size(); ++public_node) {
        Point value;
        mesh.nodes().copy_node(public_node, value.data());
        result.push_back(value);
    }
    return result;
}

ReferenceNodes make_reference_nodes(const std::vector<Point>& visible_points) {
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

void print_high_report(
    std::string_view name,
    const HighCompute::Report& report,
    const HighMesh& mesh) {
    std::cout << "    " << name << "\n"
              << "      rounds:                   " << report.rounds << '\n'
              << "      deleted vertices:         " << report.deleted_vertices << '\n'
              << "      invalidated old vertices: " << report.invalidated_old_vertices << '\n'
              << "      NEW-cell vertices:        " << report.new_cell_vertices << '\n'
              << "      repair vertices:          " << report.repair_vertices << '\n'
              << "      new reference nodes:      " << report.new_reference_nodes << '\n'
              << "      final visible nodes:      " << report.final_visible_nodes << '\n'
              << "      final internal nodes:     " << report.final_internal_nodes << '\n'
              << "      current internal slots:   " << mesh.internal_node_count() << '\n';
}

// -----------------------------------------------------------------------------
// Detect whether a visible HighVoronoi cell currently touches an invisible
// periodic copy. This is deliberately based on the actual persisted cell
// vertices rather than on geometric proximity to a periodic face.
// -----------------------------------------------------------------------------

bool visible_cell_has_invisible_generator(
    const HighMesh& owner,
    const ComputeMesh& compute_mesh,
    Index visible_cell) {
    const Index internal = owner.visible_public_to_internal(visible_cell);
    const auto compute_cell = compute_mesh.compute_public_node(internal);
    if (!compute_cell) {
        throw std::logic_error("Visible node is absent from HighVoronoiComputeMesh.");
    }

    for (const auto& vertex : compute_mesh.vertices(*compute_cell)) {
        for (const Index generator : vertex.sigma) {
            if (generator >= compute_mesh.size()) {
                continue; // boundary generator
            }

            const Index generator_internal =
                compute_mesh.stable_internal_node(generator);
            if (owner.is_active_internal(generator_internal) &&
                !owner.is_visible_internal(generator_internal)) {
                return true;
            }
        }
    }

    return false;
}

std::optional<Index> find_new_visible_cell_with_invisible_generator(
    HighMesh& owner,
    Index first_new_public,
    Index new_count) {
    ComputeRoundState inspection_state(
        owner.internal_node_count(),
        owner.internal_boundary().size(),
        owner.internal_node_count());
    ComputeMesh compute_mesh(owner, inspection_state);

    for (Index offset = Index{0}; offset < new_count; ++offset) {
        const Index public_node = static_cast<Index>(first_new_public + offset);
        if (public_node >= owner.size()) {
            break;
        }
        if (visible_cell_has_invisible_generator(owner, compute_mesh, public_node)) {
            return public_node;
        }
    }

    return std::nullopt;
}

// =============================================================================
// Compute-mesh diagnostics and independent periodic-reference comparison.
// =============================================================================

struct ComputeMeshDiagnostic {
    bool consistent = false;
    bool complete = false;
    std::size_t checked_vertex_occurrences = 0;
    std::size_t consistency_errors = 0;
    std::size_t finite_edge_endpoints = 0;
    std::size_t infinite_edges = 0;
};

ComputeMeshDiagnostic diagnose_compute_mesh(
    HighMesh& owner,
    std::string_view name) {
    ComputeRoundState inspection_state(
        owner.internal_node_count(),
        owner.internal_boundary().size(),
        owner.internal_node_count());
    ComputeMesh compute_mesh(owner, inspection_state);

    // verify_mesh_complete() includes verify_mesh() as its consistency part.
    // The completeness result is printed as a diagnostic only: HighVoronoi
    // deliberately does not persist vertices on periodic outer internal-boundary
    // planes, so global edge closure of the complete internal tiling is not by
    // itself a required invariant. Geometric consistency, however, is required.
    const auto report = highvoronoi::verify_mesh_complete(
        compute_mesh,
        VerificationTolerance,
        true);

    ComputeMeshDiagnostic result;
    result.consistent = report.consistency.valid();
    result.complete = report.complete();
    result.checked_vertex_occurrences =
        report.consistency.checked_vertex_occurrences;
    result.consistency_errors = report.consistency.error_count();
    result.finite_edge_endpoints = report.unique_finite_edge_endpoints;
    result.infinite_edges = report.infinite_edges;

    std::cout << "    compute-mesh diagnostic '" << name << "':\n"
              << "      checked vertex occurrences: "
              << result.checked_vertex_occurrences << '\n'
              << "      geometric errors:           "
              << result.consistency_errors << '\n'
              << "      finite edge endpoints:      "
              << result.finite_edge_endpoints << '\n'
              << "      infinite edges:             "
              << result.infinite_edges << '\n'
              << "      global edge closure:        "
              << (report.all_edges_have_two_occurrences ? "yes" : "NO")
              << '\n';

    const std::string consistency_description =
        std::string(name) + " compute mesh is geometrically consistent";
    check(result.consistent, consistency_description);

    return result;
}

struct ComparableVertex {
    VertexPoint position;
    NormalizedSignature signature;

    // HighVoronoi-only diagnostics. address == 0 / empty internal_signature
    // identify reference-mesh vertices.
    Address address = Address{0};
    NormalizedSignature internal_signature;
};

std::uint64_t boundary_tag(Index plane) {
    return BoundaryTagBase + static_cast<std::uint64_t>(plane);
}

NormalizedSignature normalize_high_internal_signature(
    const HighMesh& owner,
    const ComputeMesh& compute_mesh,
    const HighMesh::Sigma& sigma) {
    NormalizedSignature result;
    result.reserve(sigma.size());

    for (const Index generator : sigma) {
        if (generator < compute_mesh.size()) {
            result.push_back(static_cast<std::uint64_t>(
                compute_mesh.stable_internal_node(generator)));
            continue;
        }

        const Index plane = static_cast<Index>(generator - compute_mesh.size());
        if (plane >= owner.internal_boundary().size()) {
            throw std::logic_error("Invalid compute-mesh boundary generator.");
        }
        result.push_back(boundary_tag(plane));
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

NormalizedSignature normalize_high_signature(
    const HighMesh& owner,
    const ComputeMesh& compute_mesh,
    const HighMesh::Sigma& sigma,
    bool& touched_periodic_internal_boundary) {
    NormalizedSignature result;
    result.reserve(sigma.size());

    for (const Index generator : sigma) {
        if (generator < compute_mesh.size()) {
            const Index internal = compute_mesh.stable_internal_node(generator);
            const auto visible = owner.projected_visible_public(internal);
            if (!visible) {
                throw std::logic_error(
                    "Active compute generator has no visible HighVoronoi projection.");
            }
            result.push_back(static_cast<std::uint64_t>(*visible));
            continue;
        }

        const Index plane = static_cast<Index>(generator - compute_mesh.size());
        if (plane >= owner.internal_boundary().size()) {
            throw std::logic_error("Invalid compute-mesh boundary generator.");
        }
        if (owner.external_boundary()[plane].is_periodic()) {
            touched_periodic_internal_boundary = true;
        }
        result.push_back(boundary_tag(plane));
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

NormalizedSignature normalize_reference_signature(
    const ReferenceMesh& mesh,
    const ReferenceMesh::Sigma& sigma,
    bool& touched_outer_xy_boundary) {
    NormalizedSignature result;
    result.reserve(sigma.size());

    const Index central_count = static_cast<Index>(mesh.size() / Index{9});

    for (const Index generator : sigma) {
        if (generator < mesh.size()) {
            result.push_back(static_cast<std::uint64_t>(
                generator % central_count));
            continue;
        }

        const Index plane = static_cast<Index>(generator - mesh.size());
        if (plane >= mesh.boundary().size()) {
            throw std::logic_error("Invalid reference-mesh boundary generator.");
        }
        if (plane < Index{4}) {
            touched_outer_xy_boundary = true;
        }
        result.push_back(boundary_tag(plane));
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<ComparableVertex> collect_high_cell_vertices(
    const HighMesh& owner,
    const ComputeMesh& compute_mesh,
    Index visible_cell,
    bool& touched_periodic_internal_boundary) {
    const Index internal = owner.visible_public_to_internal(visible_cell);
    const auto compute_cell = compute_mesh.compute_public_node(internal);
    if (!compute_cell) {
        throw std::logic_error("Visible HighVoronoi node is absent from compute mesh.");
    }

    std::vector<ComparableVertex> result;
    for (const auto& vertex : compute_mesh.vertices(*compute_cell)) {
        result.push_back(ComparableVertex{
            vertex.position,
            normalize_high_signature(
                owner,
                compute_mesh,
                vertex.sigma,
                touched_periodic_internal_boundary),
            vertex.address,
            normalize_high_internal_signature(
                owner,
                compute_mesh,
                vertex.sigma)});
    }
    return result;
}

std::vector<ComparableVertex> collect_reference_cell_vertices(
    const ReferenceMesh& mesh,
    Index central_cell,
    bool& touched_outer_xy_boundary) {
    std::vector<ComparableVertex> result;
    for (const auto& vertex : mesh.vertices(central_cell)) {
        result.push_back(ComparableVertex{
            vertex.position,
            normalize_reference_signature(
                mesh,
                vertex.sigma,
                touched_outer_xy_boundary),
            Address{0},
            {}});
    }
    return result;
}

bool same_vertex(const ComparableVertex& left, const ComparableVertex& right) {
    return left.signature == right.signature &&
           (left.position - right.position).norm() <= VertexMatchTolerance;
}

bool same_internal_vertex(
    const ComparableVertex& left,
    const ComparableVertex& right) {
    return !left.internal_signature.empty() &&
           left.internal_signature == right.internal_signature &&
           (left.position - right.position).norm() <= VertexMatchTolerance;
}

std::string signature_string(const NormalizedSignature& signature) {
    std::ostringstream output;
    output << '{';
    bool first = true;
    for (const std::uint64_t value : signature) {
        if (!first) output << ',';
        first = false;
        if (value >= BoundaryTagBase) {
            output << 'B' << (value - BoundaryTagBase);
        } else {
            output << value;
        }
    }
    output << '}';
    return output.str();
}

void print_vertex(const ComparableVertex& vertex, std::string_view prefix) {
    std::cerr << "        " << prefix;
    if (vertex.address != Address{0}) {
        std::cerr << " address=" << vertex.address;
    }
    std::cerr << " position=("
              << std::setprecision(15)
              << vertex.position[0] << ", "
              << vertex.position[1] << ", "
              << vertex.position[2] << ") sigma="
              << signature_string(vertex.signature);
    if (!vertex.internal_signature.empty()) {
        std::cerr << " internal_sigma="
                  << signature_string(vertex.internal_signature);
    }
    std::cerr << '\n';
}

using NormalizedNeighbour = std::array<std::int64_t, 4>;

NormalizedNeighbour ordinary_neighbour_key(
    Index visible,
    int shift_x,
    int shift_y) {
    return NormalizedNeighbour{
        std::int64_t{0},
        static_cast<std::int64_t>(visible),
        static_cast<std::int64_t>(shift_x),
        static_cast<std::int64_t>(shift_y)};
}

NormalizedNeighbour boundary_neighbour_key(Index plane) {
    return NormalizedNeighbour{
        std::int64_t{1},
        static_cast<std::int64_t>(plane),
        std::int64_t{0},
        std::int64_t{0}};
}

std::optional<NormalizedNeighbour> normalize_high_neighbour(
    const HighMesh& owner,
    Index internal_neighbour) {
    if (internal_neighbour < owner.internal_node_count()) {
        const auto visible = owner.projected_visible_public(internal_neighbour);
        if (!visible) {
            throw std::logic_error(
                "Active HighVoronoi neighbour has no visible projection.");
        }

        Point internal_point;
        Point visible_point;
        owner.internal_nodes().copy_node(
            internal_neighbour,
            internal_point.data());
        owner.nodes().copy_node(
            *visible,
            visible_point.data());

        const Point shift = internal_point - visible_point;
        const int shift_x = static_cast<int>(std::llround(shift[0]));
        const int shift_y = static_cast<int>(std::llround(shift[1]));

        return ordinary_neighbour_key(*visible, shift_x, shift_y);
    }

    if (!owner.is_boundary_internal_index(internal_neighbour)) {
        throw std::logic_error("Invalid HighVoronoi internal neighbour index.");
    }

    const Index plane = owner.decode_boundary_internal_index(internal_neighbour);
    if (plane >= owner.external_boundary().size()) {
        throw std::logic_error("Invalid HighVoronoi neighbour boundary plane.");
    }

    // Periodic internal-boundary mirrors are construction aids, not persisted
    // physical facets of the periodic quotient mesh. They are intentionally
    // excluded from this comparison.
    if (owner.external_boundary()[plane].is_periodic()) {
        return std::nullopt;
    }

    return boundary_neighbour_key(plane);
}

std::optional<NormalizedNeighbour> normalize_reference_neighbour(
    const ReferenceMesh& mesh,
    Index neighbour) {
    const Index central_count = static_cast<Index>(mesh.size() / Index{9});

    if (neighbour < mesh.size()) {
        static constexpr std::array<std::array<int, 2>, 9> tile_shifts{{
            {{ 0,  0}},
            {{-1,  0}}, {{ 1,  0}},
            {{ 0, -1}}, {{ 0,  1}},
            {{-1, -1}}, {{-1,  1}}, {{ 1, -1}}, {{ 1,  1}}
        }};

        const Index tile = static_cast<Index>(neighbour / central_count);
        if (tile >= static_cast<Index>(tile_shifts.size())) {
            throw std::logic_error("Reference neighbour belongs to invalid tile.");
        }

        const auto& shift = tile_shifts[static_cast<std::size_t>(tile)];
        return ordinary_neighbour_key(
            static_cast<Index>(neighbour % central_count),
            shift[0],
            shift[1]);
    }

    const Index plane = static_cast<Index>(neighbour - mesh.size());
    if (plane >= mesh.boundary().size()) {
        throw std::logic_error("Invalid reference neighbour boundary plane.");
    }

    // The x/y faces are only the artificial outside of the explicit 3x3
    // tiling. Central reference cells must be compared modulo periodicity, so
    // those faces have no HighVoronoi counterpart.
    if (plane < Index{4}) {
        return std::nullopt;
    }

    return boundary_neighbour_key(plane);
}

bool compare_internal_neighbourhoods(
    HighMesh& owner,
    ReferenceMesh& reference,
    std::string_view stage_name) {
    std::vector<Index> high_raw;
    std::vector<Index> reference_raw;
    std::vector<NormalizedNeighbour> high_normalized;
    std::vector<NormalizedNeighbour> reference_normalized;

    for (Index cell = Index{0}; cell < owner.size(); ++cell) {
        const Index internal_cell = owner.visible_public_to_internal(cell);
        owner.compute_internal_neighbours(internal_cell);
        reference.compute_neighbors(cell);

        (void)owner.internal_neighbours(internal_cell, high_raw);
        const bool reference_current = reference.neighbours(cell, reference_raw);
        if (owner.internal_neighbours_dirty(internal_cell) || !reference_current) {
            std::cerr << "    [" << stage_name << ", cell " << cell
                      << "] neighbour cache stayed dirty after recomputation\n";
            return false;
        }

        high_normalized.clear();
        reference_normalized.clear();

        for (const Index neighbour : high_raw) {
            const auto normalized = normalize_high_neighbour(owner, neighbour);
            if (normalized) {
                high_normalized.push_back(*normalized);
            }
        }
        for (const Index neighbour : reference_raw) {
            const auto normalized = normalize_reference_neighbour(reference, neighbour);
            if (normalized) {
                reference_normalized.push_back(*normalized);
            }
        }

        std::sort(high_normalized.begin(), high_normalized.end());
        std::sort(reference_normalized.begin(), reference_normalized.end());

        if (high_normalized == reference_normalized) {
            continue;
        }

        auto print = [](const std::vector<NormalizedNeighbour>& values) {
            for (const auto& value : values) {
                if (value[0] == 0) {
                    std::cerr << " v" << value[1]
                              << "@(" << value[2] << ',' << value[3] << ')';
                } else {
                    std::cerr << " B" << value[1];
                }
            }
        };

        std::cerr << "    [" << stage_name << ", cell " << cell
                  << "] internal neighbour multiset mismatch\n"
                  << "      HighVoronoi:";
        print(high_normalized);
        std::cerr << "\n      reference:  ";
        print(reference_normalized);
        std::cerr << '\n';
        return false;
    }

    return true;
}

struct StageComparisonReport {
    bool all_cells_match = true;
    bool high_touched_periodic_internal_boundary = false;
    bool reference_touched_outer_xy_boundary = false;
    std::size_t total_high_vertices = 0;
    std::size_t total_reference_vertices = 0;
    std::size_t matched_vertices = 0;
    std::vector<ComparableVertex> unmatched_high_vertices;
    std::vector<ComparableVertex> all_unique_high_vertices;
};

void remember_unique_high_vertex(
    StageComparisonReport& report,
    const ComparableVertex& vertex) {
    const auto found = std::find_if(
        report.all_unique_high_vertices.begin(),
        report.all_unique_high_vertices.end(),
        [&](const ComparableVertex& existing) {
            return existing.address == vertex.address;
        });
    if (found == report.all_unique_high_vertices.end()) {
        report.all_unique_high_vertices.push_back(vertex);
    }
}

StageComparisonReport compare_current_state_with_reference(
    HighMesh& owner,
    std::string_view stage_name,
    bool print_cells = true) {
    const std::vector<Point> visible_points = collect_visible_points(owner);
    auto reference_mesh = make_reference_mesh(visible_points);

    const Index expected_reference_nodes = static_cast<Index>(
        Index{9} * static_cast<Index>(visible_points.size()));
    if (reference_mesh->size() != expected_reference_nodes) {
        throw std::logic_error("Reference tiling has unexpected node count.");
    }

    compute_reference_mesh(*reference_mesh);
    const auto reference_verification = highvoronoi::verify_mesh(
        *reference_mesh,
        VerificationTolerance,
        true);

    std::cout << "    reference '" << stage_name << "': "
              << reference_mesh->size() << " nodes, "
              << reference_verification.checked_vertex_occurrences
              << " checked occurrences, errors="
              << reference_verification.error_count() << '\n';

    const std::string reference_description =
        std::string(stage_name) + " explicit reference mesh is geometrically consistent";
    check(reference_verification.valid(), reference_description);

    ComputeRoundState inspection_state(
        owner.internal_node_count(),
        owner.internal_boundary().size(),
        owner.internal_node_count());
    ComputeMesh compute_mesh(owner, inspection_state);

    StageComparisonReport report;

    const bool neighbours_match = compare_internal_neighbourhoods(
        owner,
        *reference_mesh,
        stage_name);
    const std::string neighbour_description =
        std::string(stage_name) +
        " internal neighbour multisets match the explicit periodic reference";
    check(neighbours_match, neighbour_description);

    for (Index cell = Index{0}; cell < owner.size(); ++cell) {
        const auto high_vertices = collect_high_cell_vertices(
            owner,
            compute_mesh,
            cell,
            report.high_touched_periodic_internal_boundary);
        const auto reference_vertices = collect_reference_cell_vertices(
            *reference_mesh,
            cell,
            report.reference_touched_outer_xy_boundary);

        report.total_high_vertices += high_vertices.size();
        report.total_reference_vertices += reference_vertices.size();

        for (const ComparableVertex& vertex : high_vertices) {
            remember_unique_high_vertex(report, vertex);
        }

        std::vector<std::uint8_t> reference_used(reference_vertices.size(), 0);
        std::size_t cell_matched = 0;

        for (const ComparableVertex& high_vertex : high_vertices) {
            bool found = false;
            for (std::size_t i = 0; i < reference_vertices.size(); ++i) {
                if (reference_used[i] != 0) continue;
                if (!same_vertex(high_vertex, reference_vertices[i])) continue;

                reference_used[i] = 1;
                ++cell_matched;
                ++report.matched_vertices;
                found = true;
                break;
            }

            if (!found) {
                report.all_cells_match = false;
                report.unmatched_high_vertices.push_back(high_vertex);
                std::cerr << "    [" << stage_name << ", cell " << cell
                          << "] HighVoronoi vertex has no reference pendant:\n";
                print_vertex(high_vertex, "high");
            }
        }

        std::size_t unmatched_reference = 0;
        for (std::size_t i = 0; i < reference_vertices.size(); ++i) {
            if (reference_used[i] == 0) {
                ++unmatched_reference;
                report.all_cells_match = false;
                std::cerr << "    [" << stage_name << ", cell " << cell
                          << "] reference vertex has no HighVoronoi pendant:\n";
                print_vertex(reference_vertices[i], "reference");
            }
        }

        if (print_cells) {
            std::cout << "    cell " << std::setw(2) << cell
                      << ": HighVoronoi=" << std::setw(3) << high_vertices.size()
                      << ", reference=" << std::setw(3) << reference_vertices.size()
                      << ", matched=" << std::setw(3) << cell_matched;
            if (unmatched_reference != 0) {
                std::cout << ", unmatched reference=" << unmatched_reference;
            }
            std::cout << '\n';
        }
    }

    std::cout << "\n    stage '" << stage_name << "' totals:\n"
              << "      HighVoronoi cell-vertex occurrences: "
              << report.total_high_vertices << '\n'
              << "      reference cell-vertex occurrences:   "
              << report.total_reference_vertices << '\n'
              << "      matched occurrences:                 "
              << report.matched_vertices << '\n'
              << "      unmatched HighVoronoi occurrences:   "
              << report.unmatched_high_vertices.size() << '\n';

    return report;
}

void diagnose_final_mismatch_history(
    const StageComparisonReport& before_delete,
    const StageComparisonReport& after_delete) {
    section("6. Trace final mismatches back across the delete boundary");

    if (after_delete.unmatched_high_vertices.empty()) {
        std::cout << "    no final HighVoronoi mismatch exists to trace\n";
        return;
    }

    std::size_t same_address = 0;
    std::size_t same_internal_geometry = 0;
    std::size_t new_after_delete = 0;

    for (const ComparableVertex& final_vertex :
         after_delete.unmatched_high_vertices) {
        const auto address_match = std::find_if(
            before_delete.all_unique_high_vertices.begin(),
            before_delete.all_unique_high_vertices.end(),
            [&](const ComparableVertex& old_vertex) {
                return old_vertex.address == final_vertex.address;
            });

        if (address_match != before_delete.all_unique_high_vertices.end()) {
            ++same_address;
            std::cerr << "    [STALE ADDRESS] final mismatch already existed before delete\n";
            print_vertex(final_vertex, "after delete");
            print_vertex(*address_match, "before delete");
            continue;
        }

        const auto geometry_match = std::find_if(
            before_delete.all_unique_high_vertices.begin(),
            before_delete.all_unique_high_vertices.end(),
            [&](const ComparableVertex& old_vertex) {
                return same_internal_vertex(old_vertex, final_vertex);
            });

        if (geometry_match != before_delete.all_unique_high_vertices.end()) {
            ++same_internal_geometry;
            std::cerr << "    [RECREATED GEOMETRY] final mismatch geometry existed before delete "
                         "under a different address\n";
            print_vertex(final_vertex, "after delete");
            print_vertex(*geometry_match, "before delete");
            continue;
        }

        ++new_after_delete;
        std::cerr << "    [NEW AFTER DELETE] final mismatch was absent before delete\n";
        print_vertex(final_vertex, "after delete");
    }

    std::cout << "\n    mismatch history:\n"
              << "      same persistent address: " << same_address << '\n'
              << "      same stable geometry:     " << same_internal_geometry << '\n'
              << "      genuinely new after delete: " << new_after_delete << '\n';
}


// =============================================================================
// Targeted raw-incidence diagnostics for suspicious projected vertices.
// =============================================================================

struct RawReferenceTarget {
    NormalizedSignature projected_signature;
    VertexPoint position;
    ReferenceMesh::Sigma raw_sigma;
    ReferenceMesh::Address address = ReferenceMesh::Address{0};
};

bool same_normalized_signature(
    const NormalizedSignature& signature,
    std::initializer_list<std::uint64_t> values) {
    return signature == NormalizedSignature(values);
}

std::string reference_generator_string(
    Index generator,
    Index central_count,
    Index total_nodes) {
    if (generator >= total_nodes) {
        return "B" + std::to_string(generator - total_nodes);
    }

    static const std::array<std::pair<int, int>, 9> TileShifts{{
        { 0,  0},
        {-1,  0}, { 1,  0},
        { 0, -1}, { 0,  1},
        {-1, -1}, {-1,  1}, { 1, -1}, { 1,  1}
    }};

    const Index tile = static_cast<Index>(generator / central_count);
    const Index visible = static_cast<Index>(generator % central_count);

    std::ostringstream out;
    out << generator << "=v" << visible;
    if (tile < static_cast<Index>(TileShifts.size())) {
        const auto [sx, sy] = TileShifts[static_cast<std::size_t>(tile)];
        out << "@(" << sx << ',' << sy << ')';
    } else {
        out << "@tile" << tile;
    }
    return out.str();
}

std::string reference_raw_sigma_string(
    const ReferenceMesh::Sigma& sigma,
    Index central_count,
    Index total_nodes) {
    std::ostringstream out;
    out << '{';
    bool first = true;
    for (const Index generator : sigma) {
        if (!first) out << ", ";
        first = false;
        out << reference_generator_string(generator, central_count, total_nodes);
    }
    out << '}';
    return out.str();
}

bool reference_cell_contains_address(
    const ReferenceMesh& mesh,
    Index cell,
    ReferenceMesh::Address address) {
    for (const auto& vertex : mesh.vertices(cell)) {
        if (vertex.address == address) return true;
    }
    return false;
}

std::size_t shared_raw_generators(
    const ReferenceMesh::Sigma& left,
    const ReferenceMesh::Sigma& right) {
    std::size_t count = 0;
    for (const Index a : left) {
        if (std::find(right.begin(), right.end(), a) != right.end()) {
            ++count;
        }
    }
    return count;
}

bool has_reference_boundary_generator(
    const ReferenceMesh::Sigma& sigma,
    Index node_count) {
    return std::any_of(
        sigma.begin(), sigma.end(),
        [=](Index generator) { return generator >= node_count; });
}

std::vector<RawReferenceTarget> find_reference_targets(
    const ReferenceMesh& mesh,
    Index central_cell) {
    std::vector<RawReferenceTarget> result;
    const Index central_count = static_cast<Index>(mesh.size() / Index{9});

    for (const auto& vertex : mesh.vertices(central_cell)) {
        bool outer = false;
        const auto projected = normalize_reference_signature(
            mesh, vertex.sigma, outer);
        if (!same_normalized_signature(projected, {6, 9, 17, 26}) &&
            !same_normalized_signature(projected, {6, 12, 17, 26})) {
            continue;
        }

        result.push_back(RawReferenceTarget{
            projected,
            vertex.position,
            vertex.sigma,
            vertex.address});
    }
    return result;
}

std::string high_internal_generator_string(
    const HighMesh& owner,
    Index internal) {
    std::ostringstream out;
    if (internal >= owner.internal_node_count()) {
        if (owner.is_boundary_internal_index(internal)) {
            out << 'B' << owner.decode_boundary_internal_index(internal);
        } else {
            out << "INVALID(" << internal << ')';
        }
        return out.str();
    }

    out << internal;
    const auto visible = owner.projected_visible_public(internal);
    if (visible) out << "->v" << *visible;

    if (owner.is_visible_internal(internal)) {
        out << "[visible]";
    } else if (owner.is_reference_internal(internal)) {
        out << "[ref=";
        const auto ref = owner.reference_internal(internal);
        if (ref) out << *ref;
        else out << '?';
        out << ",shift=";
        const auto& shift = owner.reference_shift(internal);
        out << '{';
        for (std::size_t i = 0; i < shift.size(); ++i) {
            if (i != 0) out << ',';
            out << static_cast<unsigned>(shift[i]);
        }
        out << "}]";
    } else {
        out << "[invisible]";
    }
    return out.str();
}

std::string high_raw_sigma_string(
    const HighMesh& owner,
    const HighMesh::Sigma& sigma) {
    std::ostringstream out;
    out << '{';
    bool first = true;
    for (const Index generator : sigma) {
        if (!first) out << ", ";
        first = false;
        out << high_internal_generator_string(owner, generator);
    }
    out << '}';
    return out.str();
}

bool high_internal_cell_contains_address(
    const ComputeMesh& compute_mesh,
    Index internal,
    Address address) {
    const auto public_cell =
        compute_mesh.compute_public_node(internal);
    if (!public_cell) return false;

    for (const auto& vertex : compute_mesh.vertices(*public_cell)) {
        if (vertex.address == address) return true;
    }
    return false;
}

bool periodic_position_equivalent(
    const VertexPoint& high_position,
    const VertexPoint& reference_position,
    int& shift_x,
    int& shift_y,
    Scalar& residual) {
    const Scalar dx = reference_position[0] - high_position[0];
    const Scalar dy = reference_position[1] - high_position[1];
    shift_x = static_cast<int>(std::llround(dx));
    shift_y = static_cast<int>(std::llround(dy));

    VertexPoint translated = high_position;
    translated[0] += static_cast<Scalar>(shift_x);
    translated[1] += static_cast<Scalar>(shift_y);
    residual = (translated - reference_position).norm();
    return residual <= VertexMatchTolerance;
}

void diagnose_reference_target_neighbours(
    const ReferenceMesh& mesh,
    const RawReferenceTarget& target,
    Index central_count) {
    std::vector<ReferenceMesh::Address> seen;

    std::cout << "        edge-neighbours (shared raw generators >= "
              << Dimension << "):\n";

    for (const Index generator : target.raw_sigma) {
        if (generator >= mesh.size()) continue;

        for (const auto& candidate : mesh.vertices(generator)) {
            if (candidate.address == target.address) continue;
            if (std::find(seen.begin(), seen.end(), candidate.address) != seen.end()) {
                continue;
            }
            seen.push_back(candidate.address);

            const std::size_t shared =
                shared_raw_generators(target.raw_sigma, candidate.sigma);
            if (shared < static_cast<std::size_t>(Dimension)) continue;

            bool outer = false;
            const auto projected = normalize_reference_signature(
                mesh, candidate.sigma, outer);

            std::cout << "          address=" << candidate.address
                      << " shared=" << shared
                      << " position=(" << std::setprecision(15)
                      << candidate.position[0] << ", "
                      << candidate.position[1] << ", "
                      << candidate.position[2] << ")"
                      << " projected=" << signature_string(projected)
                      << " raw=" << reference_raw_sigma_string(
                             candidate.sigma, central_count, mesh.size());
            if (has_reference_boundary_generator(candidate.sigma, mesh.size())) {
                std::cout << "  [BOUNDARY PARTNER]";
            }
            std::cout << '\n';
        }
    }
}

void diagnose_target_vertex_representation(
    HighMesh& owner,
    std::string_view stage_name) {
    section(std::string("2c. Raw incidence diagnostic: ") + std::string(stage_name));

    const auto visible_points = collect_visible_points(owner);
    auto reference_mesh = make_reference_mesh(visible_points);
    compute_reference_mesh(*reference_mesh);

    const Index central_count = static_cast<Index>(visible_points.size());
    const auto targets = find_reference_targets(*reference_mesh, Index{6});

    ComputeRoundState inspection_state(
        owner.internal_node_count(),
        owner.internal_boundary().size(),
        owner.internal_node_count());
    ComputeMesh compute_mesh(owner, inspection_state);

    std::cout << "    targets found in central reference cell 6: "
              << targets.size() << '\n';

    for (const auto& target : targets) {
        std::cout << "\n    TARGET projected="
                  << signature_string(target.projected_signature)
                  << "\n        reference address=" << target.address
                  << " position=(" << std::setprecision(15)
                  << target.position[0] << ", "
                  << target.position[1] << ", "
                  << target.position[2] << ")\n"
                  << "        reference raw sigma="
                  << reference_raw_sigma_string(
                         target.raw_sigma, central_count, reference_mesh->size())
                  << '\n';

        std::cout << "        reference raw-address incidence:\n";
        for (const Index generator : target.raw_sigma) {
            if (generator >= reference_mesh->size()) continue;
            std::cout << "          "
                      << reference_generator_string(
                             generator, central_count, reference_mesh->size())
                      << " : address " << target.address
                      << (reference_cell_contains_address(
                              *reference_mesh, generator, target.address)
                              ? " PRESENT" : " MISSING")
                      << '\n';
        }

        diagnose_reference_target_neighbours(
            *reference_mesh, target, central_count);

        struct HighCandidate {
            Address address = Address{0};
            VertexPoint position;
            HighMesh::Sigma raw_sigma;
            int shift_x = 0;
            int shift_y = 0;
            Scalar residual = Scalar{0};
        };

        std::vector<HighCandidate> candidates;
        for (Index cell = Index{0}; cell < compute_mesh.size(); ++cell) {
            for (const auto& vertex : compute_mesh.vertices(cell)) {
                const auto already = std::find_if(
                    candidates.begin(), candidates.end(),
                    [&](const HighCandidate& c) {
                        return c.address == vertex.address;
                    });
                if (already != candidates.end()) continue;

                bool periodic_boundary = false;
                const auto projected = normalize_high_signature(
                    owner, compute_mesh, vertex.sigma, periodic_boundary);
                if (projected != target.projected_signature) continue;

                int sx = 0;
                int sy = 0;
                Scalar residual = Scalar{0};
                if (!periodic_position_equivalent(
                        vertex.position, target.position,
                        sx, sy, residual)) {
                    continue;
                }

                HighMesh::Sigma raw_sigma;
                VertexPoint raw_position;
                owner.database().read(vertex.address, raw_position, raw_sigma);
                if (raw_sigma.empty()) continue;

                candidates.push_back(HighCandidate{
                    vertex.address,
                    raw_position,
                    std::move(raw_sigma),
                    sx,
                    sy,
                    residual});
            }
        }

        std::cout << "        HighVoronoi period-equivalent records: "
                  << candidates.size() << '\n';

        for (const auto& candidate : candidates) {
            std::cout << "          high address=" << candidate.address
                      << " raw position=(" << candidate.position[0] << ", "
                      << candidate.position[1] << ", "
                      << candidate.position[2] << ')'
                      << " translate=(" << candidate.shift_x << ','
                      << candidate.shift_y << ",0)"
                      << " residual=" << candidate.residual << '\n'
                      << "            raw internal sigma="
                      << high_raw_sigma_string(owner, candidate.raw_sigma)
                      << '\n';

            std::cout << "            raw-internal address incidence:\n";
            for (const Index generator : candidate.raw_sigma) {
                if (generator >= owner.internal_node_count()) continue;
                std::cout << "              "
                          << high_internal_generator_string(owner, generator)
                          << " : address " << candidate.address
                          << (high_internal_cell_contains_address(
                                  compute_mesh, generator, candidate.address)
                                  ? " PRESENT" : " MISSING")
                          << '\n';
            }

            std::cout << "            canonical visible-cell incidence:\n";
            for (const std::uint64_t public64 : target.projected_signature) {
                if (public64 >= BoundaryTagBase) continue;
                const Index public_node = static_cast<Index>(public64);
                const Index internal = owner.visible_public_to_internal(public_node);
                std::cout << "              visible " << public_node
                          << " -> internal " << internal
                          << " : address " << candidate.address
                          << (high_internal_cell_contains_address(
                                  compute_mesh, internal, candidate.address)
                                  ? " PRESENT" : " absent")
                          << '\n';
            }
        }
    }
}

} // namespace

int main() {
    section("HighVoronoi incremental periodic-reference regression");
    std::cout
        << "Visible domain: [0,1]^3\n"
        << "Periodic axes:  x (0), y (1)\n"
        << "Non-periodic:   z (2)\n"
        << "Workflow:       28 -> compute -> +3 -> compute -> delete 1 -> compute\n"
        << "Diagnostics:    compute-mesh consistency and explicit reference both\n"
        << "                before and after the delete\n";

    // -------------------------------------------------------------------------
    // 1. Initial periodic HighVoronoi construction from 28 visible nodes.
    // -------------------------------------------------------------------------
    section("1. Initial compute from 28 visible nodes");

    const std::vector<Point> initial_points = make_initial_visible_points();
    auto high_mesh = make_high_mesh(initial_points);

    check(high_mesh->size() == InitialVisibleNodeCount,
          "HighVoronoi starts with exactly 28 visible nodes");

    HighCompute initial_compute(
        *high_mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters(),
        EdgeParameters{highvoronoi::DirectHash{HashCapacity}});

    const auto initial_report = initial_compute.compute();
    print_high_report("initial compute", initial_report, *high_mesh);

    check(high_mesh->size() == InitialVisibleNodeCount,
          "periodization keeps the public node count at 28");
    check(high_mesh->internal_node_count() > high_mesh->visible_public_count(),
          "initial compute created invisible periodic reference nodes");
    check(high_mesh->integrated_node_count() == high_mesh->internal_node_count(),
          "initial periodic state is fully integrated");

    // -------------------------------------------------------------------------
    // 2. Insert three separated visible generators near periodic faces.
    // -------------------------------------------------------------------------
    section("2. Insert three boundary-near visible nodes and refine");

    const std::vector<Point> added_points = refinement_points();
    const Index first_added_public = high_mesh->size();
    append_visible_points(*high_mesh, added_points);

    check(high_mesh->size() == Index{31},
          "three new visible nodes were appended after the initial 28");

    HighCompute refinement_compute(
        *high_mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters(),
        EdgeParameters{highvoronoi::DirectHash{HashCapacity}});

    const auto refinement_report = refinement_compute.compute();
    print_high_report("after +3 refinement", refinement_report, *high_mesh);

    check(high_mesh->size() == Index{31},
          "periodic refinement keeps all 31 visible nodes public");
    check(high_mesh->integrated_node_count() == high_mesh->internal_node_count(),
          "refined periodic state is fully integrated");

    section("2a. Validate actual HighVoronoiComputeMesh after +3 refinement");
    const auto before_delete_compute_diagnostic =
        diagnose_compute_mesh(*high_mesh, "after +3 refinement");
    (void)before_delete_compute_diagnostic;

    section("2b. Compare the 31-node refined state with an independent reference");
    const StageComparisonReport before_delete_reference =
        compare_current_state_with_reference(*high_mesh, "before delete", true);

    check(!before_delete_reference.high_touched_periodic_internal_boundary,
          "31-node visible cells contain no persisted periodic-boundary vertices");
    check(!before_delete_reference.reference_touched_outer_xy_boundary,
          "31-node central reference cells do not reach artificial outer x/y faces");
    check(before_delete_reference.total_high_vertices ==
              before_delete_reference.total_reference_vertices,
          "31-node HighVoronoi and reference contain the same cell-vertex count");
    check(before_delete_reference.all_cells_match,
          "31-node state matches the one-shot explicit periodic reference");

    diagnose_target_vertex_representation(*high_mesh, "before delete");

    // -------------------------------------------------------------------------
    // 3. Delete one of the newly inserted nodes that actually sees an invisible
    //    periodic generator.
    // -------------------------------------------------------------------------
    section("3. Delete one newly added node with an invisible neighbour");

    const auto delete_public = find_new_visible_cell_with_invisible_generator(
        *high_mesh,
        first_added_public,
        AddedVisibleNodeCount);

    check(delete_public.has_value(),
          "at least one of the three inserted boundary-near nodes has an invisible neighbour");

    if (!delete_public) {
        section("Result");
        std::cout << "performed checks: " << performed_checks << '\n'
                  << "failed checks:    " << failed_checks << '\n';
        return 1;
    }

    std::cout << "    deleting visible public node " << *delete_public << '\n';

    const auto removed = high_mesh->erase_visible_nodes(
        std::vector<Index>{*delete_public});

    check(removed.removed_internal.size() >= 1,
          "deletion removes the visible node and records its internal deletion");
    check(high_mesh->size() == FinalVisibleNodeCount,
          "exactly 30 visible nodes remain before delete repair");

    HighCompute delete_compute(
        *high_mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters(),
        EdgeParameters{highvoronoi::DirectHash{HashCapacity}});

    const auto delete_report = delete_compute.compute();
    print_high_report("after delete repair", delete_report, *high_mesh);

    check(high_mesh->size() == FinalVisibleNodeCount,
          "delete repair leaves exactly 30 visible nodes");
    check(high_mesh->integrated_node_count() == high_mesh->internal_node_count(),
          "final HighVoronoi state is fully integrated");

    section("3a. Validate actual HighVoronoiComputeMesh after delete repair");
    const auto after_delete_compute_diagnostic =
        diagnose_compute_mesh(*high_mesh, "after delete repair");
    (void)after_delete_compute_diagnostic;

    // -------------------------------------------------------------------------
    // 4. Independent final-state reference and comparison.
    // -------------------------------------------------------------------------
    section("4. Compare final 30-node state with an independent reference");

    const StageComparisonReport after_delete_reference =
        compare_current_state_with_reference(*high_mesh, "after delete", true);

    check(!after_delete_reference.high_touched_periodic_internal_boundary,
          "final visible HighVoronoi cells contain no persisted periodic-boundary vertices");
    check(!after_delete_reference.reference_touched_outer_xy_boundary,
          "final central reference cells do not reach artificial outer x/y faces");
    check(after_delete_reference.total_high_vertices ==
              after_delete_reference.total_reference_vertices,
          "final HighVoronoi and reference contain the same cell-vertex count");
    check(after_delete_reference.all_cells_match,
          "final state matches the one-shot explicit periodic reference");

    // -------------------------------------------------------------------------
    // 5. Determine whether any final bad vertex is stale across the delete.
    // -------------------------------------------------------------------------
    diagnose_final_mismatch_history(
        before_delete_reference,
        after_delete_reference);

    section("Result");
    std::cout << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}





