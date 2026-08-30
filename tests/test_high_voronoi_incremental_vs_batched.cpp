#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/edge_iterator.hpp>
#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters>;
using Mesh = highvoronoi::VoronoiMesh<
    Scalar,
    Dimension,
    Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Sigma = Mesh::Sigma;
using Boundary = Mesh::BoundaryType;

using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::InRangeRaycast,
    Scalar>;

inline constexpr std::size_t StoredUnits = 65536;
inline constexpr std::size_t HashCapacity = 65536;
inline constexpr Scalar VerificationTolerance = Scalar{1e-14};
inline constexpr Scalar RayVarianceTolerance = Scalar{9e-14};

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

Boundary boundary_minus_one_plus_one() {
    return Boundary::cuboid(
        point(2.0, 2.0, 2.0),
        point(-1.0, -1.0, -1.0),
        std::vector<Index>{});
}

Boundary unit_cube_boundary() {
    return Boundary::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{});
}

DatabaseParameters database_parameters(std::size_t capacity = HashCapacity) {
    return DatabaseParameters{highvoronoi::DirectHash{capacity}};
}

EdgeParameters edge_parameters(std::size_t capacity = HashCapacity) {
    return EdgeParameters{highvoronoi::DirectHash{capacity}};
}

RayParameters ray_parameters() {
    RayParameters parameters;
    parameters.variance_tolerance = RayVarianceTolerance;
    return parameters;
}

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        StoredUnits,
        database_parameters());
}

std::unique_ptr<Mesh> make_mesh(
    const std::vector<Point>& points,
    const Boundary& boundary) {
    Nodes nodes(static_cast<Index>(points.size()));
    for (Index i = Index{0}; i < static_cast<Index>(points.size()); ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
    }
    return std::make_unique<Mesh>(
        std::move(nodes),
        boundary,
        make_database());
}

std::size_t unique_primary_vertex_count(const Mesh& mesh) {
    std::unordered_set<Mesh::Address> addresses;
    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        for (const auto& vertex : mesh.primary_vertices(cell)) {
            addresses.insert(vertex.address);
        }
    }
    return addresses.size();
}

void print_sigma(const Sigma& sigma) {
    std::cout << '{';
    for (std::size_t i = 0; i < sigma.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << sigma[i];
    }
    std::cout << '}';
}

void print_index_set(const std::set<Index>& values) {
    std::cout << '{';
    bool first = true;
    for (const Index value : values) {
        if (!first) {
            std::cout << ',';
        }
        std::cout << value;
        first = false;
    }
    std::cout << '}';
}

// -----------------------------------------------------------------------------
// Independent cell-local edge diagnostic.
//
// It mirrors the intended completeness invariant without using EdgeHashTable:
// for each cell, each finite complete supporting edge must be represented by
// exactly two different global skip generators. Raw repetitions are retained
// separately so we can distinguish a missing endpoint from duplicate local
// representations of a degenerate edge.
// -----------------------------------------------------------------------------

struct EdgeOccurrenceSummary {
    std::set<Index> skips;
    std::size_t raw_occurrences = 0;
};

struct EdgeDiagnosticReport {
    std::size_t cells_with_incomplete_edges = 0;
    std::size_t incomplete_edges = 0;
    std::size_t overcomplete_edges = 0;
};

EdgeDiagnosticReport diagnose_cell_local_edges(
    Mesh& mesh,
    std::size_t maximum_details = 24) {
    using Context = highvoronoi::detail::MeshValidationContext<Mesh>;
    using ExtendedNodes = std::remove_reference_t<
        decltype(std::declval<Context&>().extended_nodes())>;
    using Iterator = highvoronoi::EdgeIterator<
        ExtendedNodes,
        highvoronoi::detail::EmptyLock>;

    Context context(mesh);
    Iterator iterator(context.extended_nodes());

    EdgeDiagnosticReport report;
    std::size_t printed = 0;

    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        context.activate_cell(cell);

        std::map<Sigma, EdgeOccurrenceSummary> edges;

        for (const auto& vertex : mesh.vertices(cell)) {
            iterator.reset(
                vertex.sigma,
                vertex.position,
                cell,
                typename Iterator::OnQueueEdges{});

            while (const auto edge = iterator.next()) {
                Sigma full_edge(
                    edge->full_indices().begin(),
                    edge->full_indices().end());
                std::sort(full_edge.begin(), full_edge.end());

                auto& summary = edges[full_edge];
                summary.skips.insert(edge->skip());
                ++summary.raw_occurrences;
            }
        }

        bool cell_incomplete = false;
        for (const auto& [full_edge, summary] : edges) {
            if (summary.skips.size() == 2) {
                continue;
            }

            cell_incomplete = true;
            if (summary.skips.size() < 2) {
                ++report.incomplete_edges;
            } else {
                ++report.overcomplete_edges;
            }

            if (printed < maximum_details) {
                std::cout << "      cell=" << cell << " full_edge=";
                print_sigma(full_edge);
                std::cout << " skips=";
                print_index_set(summary.skips);
                std::cout << " raw occurrences=" << summary.raw_occurrences
                          << '\n';
                ++printed;
            }
        }

        if (cell_incomplete) {
            ++report.cells_with_incomplete_edges;
        }
    }

    if (report.incomplete_edges + report.overcomplete_edges > printed) {
        std::cout << "      ... "
                  << (report.incomplete_edges + report.overcomplete_edges - printed)
                  << " further problematic cell/edge pairs suppressed\n";
    }

    return report;
}

struct CaseResult {
    bool compute_finished = false;
    bool consistent = false;
    bool complete = false;
    std::size_t node_count = 0;
    std::size_t primary_vertices = 0;
    std::size_t new_vertices = 0;
    EdgeDiagnosticReport edge_diagnostic;
};

CaseResult run_case(
    std::string_view name,
    const std::vector<Point>& points,
    const Boundary& boundary,
    std::size_t maximum_edge_details = 24) {
    section(name);
    std::cout << "    nodes = " << points.size() << '\n'
              << "    raycaster = InRangeRaycast\n";

    auto mesh = make_mesh(points, boundary);

    auto tree = highvoronoi::geometry::make_search_tree(
        *mesh,
        highvoronoi::geometry::KDSearch{8, 1});
    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters());

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        DatabaseParameters,
        EdgeParameters>;

    Compute compute(
        *mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,
        database_parameters(16384),
        edge_parameters(65536));

    CaseResult result;
    result.node_count = points.size();

    try {
        compute.compute();
        result.compute_finished = true;
        result.new_vertices = compute.new_vertex_count();
    } catch (const std::exception& error) {
        std::cerr << "    ComputeVoronoi threw: " << error.what() << '\n';
    }

    check(result.compute_finished, "ComputeVoronoi finishes");
    if (!result.compute_finished) {
        return result;
    }

    result.primary_vertices = unique_primary_vertex_count(*mesh);

    const auto consistency = highvoronoi::verify_mesh(
        *mesh,
        VerificationTolerance,
        true);
    const auto completeness = highvoronoi::verify_mesh_complete(
        *mesh,
        VerificationTolerance,
        true);

    result.consistent = consistency.valid();
    result.complete = completeness.complete();

    std::cout << "    compute.new_vertex_count = " << result.new_vertices << '\n'
              << "    unique primary vertices  = " << result.primary_vertices << '\n'
              << "    checked occurrences      = "
              << consistency.checked_vertex_occurrences << '\n'
              << "    geometric errors         = " << consistency.error_count() << '\n'
              << "    finite edge endpoints    = "
              << completeness.unique_finite_edge_endpoints << '\n'
              << "    edge closure             = "
              << (completeness.all_edges_have_two_occurrences ? "yes" : "NO")
              << '\n';

    check(result.consistent, "mesh is geometrically consistent");

    if (!result.complete) {
        std::cout << "    [edge diagnostic] problematic cell-local full edges:\n";
        result.edge_diagnostic = diagnose_cell_local_edges(
            *mesh,
            maximum_edge_details);
        std::cout << "      cells with incomplete edges = "
                  << result.edge_diagnostic.cells_with_incomplete_edges << '\n'
                  << "      edges with <2 skips          = "
                  << result.edge_diagnostic.incomplete_edges << '\n'
                  << "      edges with >2 skips          = "
                  << result.edge_diagnostic.overcomplete_edges << '\n';
    }

    check(result.complete, "mesh is topologically complete");
    return result;
}

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

std::vector<Point> make_background_points() {
    std::vector<Point> points;
    points.reserve(10);

    std::mt19937_64 random(0x4856454e47494e45ULL);
    std::uniform_real_distribution<Scalar> coordinate(-0.90, -0.10);
    while (points.size() < 9) {
        const Point candidate = point(
            coordinate(random),
            coordinate(random),
            coordinate(random));

        bool separated = true;
        for (const Point& existing : points) {
            if ((candidate - existing).norm() < Scalar{0.10}) {
                separated = false;
                break;
            }
        }
        if (separated) {
            points.push_back(candidate);
        }
    }

    points.push_back(point(0.53, 0.47, 0.56));
    return points;
}

std::vector<Point> make_cartesian_grid(
    Index side,
    Scalar start,
    Scalar spacing) {
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(side * side * side));

    for (Index z = Index{0}; z < side; ++z) {
        for (Index y = Index{0}; y < side; ++y) {
            for (Index x = Index{0}; x < side; ++x) {
                points.push_back(point(
                    start + spacing * static_cast<Scalar>(x),
                    start + spacing * static_cast<Scalar>(y),
                    start + spacing * static_cast<Scalar>(z)));
            }
        }
    }
    return points;
}

std::vector<Point> perturb_points(
    const std::vector<Point>& input,
    std::uint64_t seed,
    Scalar amplitude = Scalar{0.001}) {
    std::vector<Point> result = input;
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<Scalar> perturb(-amplitude, amplitude);

    for (Point& value : result) {
        for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
            value[coordinate] += perturb(random);
        }
    }
    return result;
}

Index flattened_grid_node(Index side, Index x, Index y, Index z) {
    return x + side * y + side * side * z;
}

std::vector<Point> erase_indices(
    const std::vector<Point>& input,
    std::vector<std::size_t> indices) {
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    std::vector<Point> result;
    result.reserve(input.size() - indices.size());

    std::size_t remove_position = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (remove_position < indices.size() &&
            i == indices[remove_position]) {
            ++remove_position;
            continue;
        }
        result.push_back(input[i]);
    }
    return result;
}

// Exact final node set from diagnostic D:
//   9 of the original 10 background nodes (background index 2 removed),
//   plus 998 of the 10x10x10 grid nodes (5,5,5) and (7,7,7) removed.
std::vector<Point> make_final_d_nodes(bool perturb_grid) {
    auto background = make_background_points();
    background.erase(background.begin() + 2);

    auto grid = make_cartesian_grid(Index{10}, Scalar{0.05}, Scalar{0.10});
    grid = erase_indices(
        grid,
        {
            static_cast<std::size_t>(flattened_grid_node(10, 5, 5, 5)),
            static_cast<std::size_t>(flattened_grid_node(10, 7, 7, 7))
        });

    if (perturb_grid) {
        grid = perturb_points(grid, 0x4450455254555242ULL);
    }

    background.insert(background.end(), grid.begin(), grid.end());
    return background;
}

std::vector<Point> make_small_extra_points() {
    return {
        point(-0.78, -0.62, -0.51),
        point(-0.63, -0.31, -0.74),
        point(-0.35, -0.81, -0.28),
        point(-0.22, -0.44, -0.68),
        point(0.53, 0.47, 0.56),
        point(0.86, -0.37, 0.18)
    };
}

std::vector<Point> make_small_2x2x2_plus_extras(bool perturb_grid) {
    auto result = make_small_extra_points();
    auto grid = make_cartesian_grid(Index{2}, Scalar{0.35}, Scalar{0.30});
    if (perturb_grid) {
        grid = perturb_points(grid, 0x534d414c4c504552ULL);
    }
    result.insert(result.end(), grid.begin(), grid.end());
    return result;
}

std::vector<Point> make_4x4x4_missing_two(bool perturb_grid) {
    auto grid = make_cartesian_grid(Index{4}, Scalar{0.20}, Scalar{0.20});

    grid = erase_indices(
        grid,
        {
            static_cast<std::size_t>(flattened_grid_node(4, 1, 1, 1)),
            static_cast<std::size_t>(flattened_grid_node(4, 2, 2, 2))
        });

    if (perturb_grid) {
        grid = perturb_points(grid, 0x3447524944504552ULL);
    }
    return grid;
}

} // namespace

int main() {
    section("ComputeVoronoi mixed Cartesian diagnostics");
    std::cout
        << "Every construction uses an ordinary VoronoiMesh and explicit "
        << "InRangeRaycast.\n"
        << "Perturbed Cartesian nodes receive independent coordinate noise in "
        << "[-0.001,+0.001].\n";

    // Control: the known-good exact Cartesian case.
    run_case(
        "0. CONTROL: exact 4x4x4 Cartesian grid",
        make_cartesian_grid(Index{4}, Scalar{0.20}, Scalar{0.20}),
        unit_cube_boundary());

    // D reproduced without HighVoronoi or CuboidMeshEngine.
    run_case(
        "1. EXACT D: final 1007-node set from the HighVoronoi diagnostic",
        make_final_d_nodes(false),
        boundary_minus_one_plus_one(),
        40);

    // Same D node topology, but destroy Cartesian degeneracy only in grid nodes.
    run_case(
        "2. PERTURBED D: same final node set, Cartesian grid perturbed",
        make_final_d_nodes(true),
        boundary_minus_one_plus_one(),
        40);

    // Small analogue: 2x2x2 exact cube plus six additional nodes.
    run_case(
        "3. SMALL EXACT: 2x2x2 Cartesian grid + six extra nodes",
        make_small_2x2x2_plus_extras(false),
        boundary_minus_one_plus_one(),
        40);

    run_case(
        "4. SMALL PERTURBED: 2x2x2 grid perturbed + six extra nodes",
        make_small_2x2x2_plus_extras(true),
        boundary_minus_one_plus_one(),
        40);

    // Pure grid with two holes: isolates missing generators from foreign nodes.
    run_case(
        "5. PURE EXACT WITH HOLES: 4x4x4 grid minus nodes (1,1,1) and (2,2,2)",
        make_4x4x4_missing_two(false),
        unit_cube_boundary(),
        40);

    run_case(
        "6. PURE PERTURBED WITH HOLES: same 4x4x4 grid, perturbed",
        make_4x4x4_missing_two(true),
        unit_cube_boundary(),
        40);

    section("Result");
    std::cout << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}
