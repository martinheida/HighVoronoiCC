
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/high_voronoi/compute_high_voronoi.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <iterator>
#include <set>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;
inline constexpr Index InitialVisibleNodeCount = Index{28};
inline constexpr Index AddedVisibleNodeCount = Index{3};

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, DatabaseParameters, Dimension>;

using HighMesh = highvoronoi::HighVoronoiMesh<
    Scalar,
    Dimension,
    Database>;
using Point = HighMesh::NodePoint;
using Boundary = HighMesh::BoundaryType;

using OrdinaryMesh = highvoronoi::VoronoiMesh<
    Scalar,
    Dimension,
    Database>;
using OrdinaryNodes = OrdinaryMesh::InternalNodes;

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

using Refine = highvoronoi::RefineVoronoi<
    OrdinaryMesh,
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

EdgeParameters edge_parameters() {
    return EdgeParameters{highvoronoi::DirectHash{HashCapacity}};
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
    return Boundary::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{Index{0}, Index{1}});
}

Boundary large_fixed_replay_boundary() {
    // This is the fixed 3x3 periodic-reference box already used by the
    // historical HighVoronoi regression: [-1,2] x [-1,2] x [0,1].
    // It is deliberately independent of HighVoronoi's changing internal
    // boundary, and all step-2 generator copies lie strictly inside it.
    return Boundary::cuboid(
        point(3.0, 3.0, 1.0),
        point(-1.0, -1.0, 0.0),
        std::vector<Index>{});
}

std::vector<Point> refinement_points() {
    return {
        point(0.008, 0.18, 0.37),
        point(0.992, 0.73, 0.63),
        point(0.43, 0.009, 0.84)
    };
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

std::vector<Point> make_initial_visible_points() {
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(InitialVisibleNodeCount));

    const std::vector<Point> future = refinement_points();
    std::mt19937_64 random(0x494e4352454d454eULL); // "INCREMEN"
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{0.025},
        Scalar{0.975});

    while (points.size() < static_cast<std::size_t>(InitialVisibleNodeCount)) {
        const Point candidate = point(
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

std::unique_ptr<HighMesh> make_high_mesh() {
    auto mesh = std::make_unique<HighMesh>(
        Index{Dimension},
        periodic_unit_cube(),
        std::in_place,
        DatabaseUnits,
        database_parameters());

    for (const Point& value : make_initial_visible_points()) {
        (void)mesh->append_visible_node(value);
    }
    return mesh;
}

HighCompute::Report run_high_compute(HighMesh& mesh) {
    HighCompute compute(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters(),
        edge_parameters());
    return compute.compute();
}

Point high_internal_point(const HighMesh& mesh, Index internal) {
    Point result;
    mesh.internal_nodes().copy_node(internal, result.data());
    return result;
}

std::vector<Point> collect_high_internal_points(const HighMesh& mesh) {
    std::vector<Point> result;
    result.reserve(static_cast<std::size_t>(mesh.internal_node_count()));
    for (Index internal = Index{0};
         internal < mesh.internal_node_count();
         ++internal) {
        if (!mesh.is_active_internal(internal)) {
            throw std::logic_error(
                "Replay diagnostic expects no inactive internal nodes before delete.");
        }
        result.push_back(high_internal_point(mesh, internal));
    }
    return result;
}

std::string shift_mask_string(const HighMesh::ShiftMask& mask) {
    std::ostringstream output;
    output << '{';
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << static_cast<int>(mask[i]);
    }
    output << '}';
    return output.str();
}

void print_high_node(const HighMesh& mesh, Index internal, std::string_view origin) {
    const Point p = high_internal_point(mesh, internal);
    const auto reference = mesh.reference_internal(internal);

    std::cout << "      internal=" << std::setw(3) << internal
              << "  source=" << std::setw(18) << origin
              << "  visible=" << (mesh.is_visible_internal(internal) ? 1 : 0)
              << "  reference=";

    if (reference) {
        std::cout << *reference;
    } else {
        std::cout << '-';
    }

    std::cout << "  shift=" << shift_mask_string(mesh.reference_shift(internal))
              << "  x=(" << std::setprecision(17)
              << p[0] << ',' << p[1] << ',' << p[2] << ")\n";
}

bool all_points_inside_boundary(
    const std::vector<Point>& points,
    const Boundary& boundary) {
    for (const Point& p : points) {
        if (!boundary.contains(p, Scalar{1e-12})) {
            std::cerr << "      point outside replay boundary: ("
                      << std::setprecision(17)
                      << p[0] << ',' << p[1] << ',' << p[2] << ")\n";
            return false;
        }
    }
    return true;
}

std::unique_ptr<OrdinaryMesh> make_ordinary_mesh(
    const std::vector<Point>& points,
    const Boundary& boundary) {
    OrdinaryNodes nodes(static_cast<Index>(points.size()));
    for (Index i = Index{0}; i < static_cast<Index>(points.size()); ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
    }

    return std::make_unique<OrdinaryMesh>(
        std::move(nodes),
        boundary,
        make_database());
}

void compute_ordinary_full(OrdinaryMesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});
    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters());

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        OrdinaryMesh,
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
        edge_parameters());
    compute.compute();
}

using SignatureSet = std::set<std::vector<Index>>;

SignatureSet finite_vertex_signatures(const OrdinaryMesh& mesh) {
    SignatureSet signatures;

    // Every finite record occurs exactly once in a primary address list. Using
    // primary_vertices therefore compares persisted finite topology without
    // counting the secondary per-cell registrations of the same vertex.
    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        for (const auto& vertex : mesh.primary_vertices(cell)) {
            std::vector<Index> sigma(vertex.sigma.begin(), vertex.sigma.end());
            std::sort(sigma.begin(), sigma.end());
            signatures.insert(std::move(sigma));
        }
    }

    return signatures;
}

std::vector<std::vector<Index>> set_difference(
    const SignatureSet& left,
    const SignatureSet& right) {
    std::vector<std::vector<Index>> result;
    std::set_difference(
        left.begin(), left.end(),
        right.begin(), right.end(),
        std::back_inserter(result));
    return result;
}

void print_sigma(const std::vector<Index>& sigma) {
    std::cout << '{';
    for (std::size_t i = 0; i < sigma.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << sigma[i];
    }
    std::cout << '}';
}

void print_signature_difference(
    std::string_view label,
    const std::vector<std::vector<Index>>& signatures,
    std::size_t maximum_printed = 20) {
    std::cout << "      " << label << ": " << signatures.size() << '\n';
    const std::size_t shown = std::min(maximum_printed, signatures.size());
    for (std::size_t i = 0; i < shown; ++i) {
        std::cout << "        ";
        print_sigma(signatures[i]);
        std::cout << '\n';
    }
    if (shown < signatures.size()) {
        std::cout << "        ... " << (signatures.size() - shown)
                  << " more\n";
    }
}

bool validate_refinement_state(
    OrdinaryMesh& incremental,
    const std::vector<Point>& points,
    const Boundary& boundary,
    std::size_t block_number) {
    auto reference = make_ordinary_mesh(points, boundary);
    compute_ordinary_full(*reference);

    const auto incremental_complete = highvoronoi::verify_mesh_complete(
        incremental,
        VerificationTolerance,
        false);
    const auto reference_complete = highvoronoi::verify_mesh_complete(
        *reference,
        VerificationTolerance,
        false);

    const SignatureSet incremental_signatures =
        finite_vertex_signatures(incremental);
    const SignatureSet reference_signatures =
        finite_vertex_signatures(*reference);

    // EXTRA is precisely the failure mode we are looking for: a finite vertex
    // persisted by incremental refinement although the one-shot diagram with
    // exactly the same generators does not contain that signature.
    const auto extra = set_difference(
        incremental_signatures,
        reference_signatures);
    const auto missing = set_difference(
        reference_signatures,
        incremental_signatures);

    std::cout << "\n      state after block " << block_number << ":\n"
              << "        incremental consistency: "
              << (incremental_complete.consistency.valid() ? "YES" : "NO") << '\n'
              << "        incremental complete:    "
              << (incremental_complete.complete() ? "YES" : "NO") << '\n'
              << "        reference consistency:   "
              << (reference_complete.consistency.valid() ? "YES" : "NO") << '\n'
              << "        reference complete:      "
              << (reference_complete.complete() ? "YES" : "NO") << '\n'
              << "        incremental signatures:  "
              << incremental_signatures.size() << '\n'
              << "        reference signatures:    "
              << reference_signatures.size() << '\n'
              << "        EXTRA signatures:        " << extra.size() << '\n'
              << "        MISSING signatures:      " << missing.size() << '\n';

    if (!extra.empty()) {
        print_signature_difference(
            "EXTRA persisted finite vertices",
            extra);
    }
    if (!missing.empty()) {
        print_signature_difference(
            "MISSING finite vertices",
            missing);
    }

    const bool reference_ok =
        reference_complete.consistency.valid() &&
        reference_complete.complete();

    const bool incremental_ok =
        incremental_complete.consistency.valid() &&
        incremental_complete.complete() &&
        extra.empty() &&
        missing.empty();

    if (!reference_ok) {
        std::cerr << "\n    [FAIL] one-shot reference itself is invalid after block "
                  << block_number << "; this test state cannot be interpreted.\n";
        (void)highvoronoi::verify_mesh_complete(
            *reference,
            VerificationTolerance,
            true);
        return false;
    }

    if (incremental_ok) {
        std::cout << "    [OK]   ordinary refinement remains consistent and complete "
                  << "after NEW block " << block_number << '\n';
        return true;
    }

    std::cerr << "\n    [FAIL] ordinary refinement has a topological/geometric "
              << "regression after NEW block " << block_number << '\n';

    // Detailed geometric diagnostics are especially useful for stale vertices:
    // verify_mesh_complete prints the nearest generator that invalidates sigma.
    if (!incremental_complete.consistency.valid() ||
        !incremental_complete.complete()) {
        (void)highvoronoi::verify_mesh_complete(
            incremental,
            VerificationTolerance,
            true);
    }

    return false;
}

} // namespace

int main() {
    section("HighVoronoi step-2 NEW-block replay into ordinary VoronoiMesh");
    std::cout
        << "Historical geometry: 28 visible nodes in [0,1]^3\n"
        << "Periodic axes: x, y\n"
        << "Step 2 visible inserts: +3\n"
        << "Replay rule: preserve exactly the HighVoronoi NEW blocks.\n"
        << "Ordinary replay domain: [-1,2] x [-1,2] x [0,1], fixed and non-periodic.\n"
        << "Only the generator blocks are inherited from HighVoronoi.\n";

    // ---------------------------------------------------------------------
    // A. Run the historical HighVoronoi construction and record exactly the
    //    NEW frontier used in every step-2 outer round.
    // ---------------------------------------------------------------------
    section("A. Generate the historical HighVoronoi step-2 NEW blocks");

    auto high_mesh = make_high_mesh();
    check(high_mesh->size() == InitialVisibleNodeCount,
          "HighVoronoi starts with 28 visible nodes");

    const auto initial_report = run_high_compute(*high_mesh);
    (void)initial_report;

    const Index baseline_internal_count = high_mesh->internal_node_count();
    const std::vector<Point> baseline_points = collect_high_internal_points(*high_mesh);

    std::cout << "    after initial compute: visible=" << high_mesh->size()
              << " internal=" << baseline_internal_count << '\n';

    for (const Point& p : refinement_points()) {
        (void)high_mesh->append_visible_node(p);
    }

    const Index after_visible_append_count = high_mesh->internal_node_count();
    check(after_visible_append_count ==
              static_cast<Index>(baseline_internal_count + AddedVisibleNodeCount),
          "the three visible step-2 nodes form the first NEW block");

    const auto step2_report = run_high_compute(*high_mesh);
    const Index final_internal_count = high_mesh->internal_node_count();
    const std::vector<Point> final_points = collect_high_internal_points(*high_mesh);

    check(step2_report.new_node_blocks.size() == step2_report.rounds,
          "HighVoronoi report contains one NEW block for every outer round");

    std::cout << "\n    HighVoronoi step-2 NEW blocks:\n";
    for (std::size_t block = 0; block < step2_report.new_node_blocks.size(); ++block) {
        const auto& indices = step2_report.new_node_blocks[block];
        std::cout << "\n      block " << (block + 1)
                  << "  count=" << indices.size() << "  internal={";
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << indices[i];
        }
        std::cout << "}\n";

        for (const Index internal : indices) {
            print_high_node(
                *high_mesh,
                internal,
                block == 0 ? "visible +3" : "periodic reference");
        }
    }

    std::size_t recorded_new_nodes = 0;
    for (const auto& block : step2_report.new_node_blocks) {
        recorded_new_nodes += block.size();
    }

    check(recorded_new_nodes ==
              static_cast<std::size_t>(final_internal_count - baseline_internal_count),
          "NEW blocks contain every node added during step 2 exactly once");

    const Boundary replay_boundary = large_fixed_replay_boundary();

    check(all_points_inside_boundary(final_points, replay_boundary),
          "all HighVoronoi step-2 generators lie inside the fixed replay box");

    // ---------------------------------------------------------------------
    // B. Start a plain VoronoiMesh from exactly the pre-step-2 internal nodes
    //    in one fixed large non-periodic box. Add each HighVoronoi NEW frontier
    //    as one RefineVoronoi block. The question is whether stale vertices
    //    survive this same refinement sequence.
    // ---------------------------------------------------------------------
    section("B. Replay the same NEW blocks with RefineVoronoi");

    std::vector<Point> replay_points = baseline_points;
    auto incremental = make_ordinary_mesh(replay_points, replay_boundary);
    compute_ordinary_full(*incremental);

    const auto initial_complete = highvoronoi::verify_mesh_complete(
        *incremental,
        VerificationTolerance,
        false);
    check(initial_complete.consistency.valid() && initial_complete.complete(),
          "ordinary baseline is consistent and complete in the fixed replay box");

    bool all_blocks_ok =
        initial_complete.consistency.valid() && initial_complete.complete();

    for (std::size_t block = 0; block < step2_report.new_node_blocks.size(); ++block) {
        const auto& internal_indices = step2_report.new_node_blocks[block];
        std::vector<Point> block_points;
        block_points.reserve(internal_indices.size());

        for (const Index internal : internal_indices) {
            block_points.push_back(final_points[static_cast<std::size_t>(internal)]);
        }

        std::cout << "\n    RefineVoronoi block " << (block + 1)
                  << " with " << block_points.size() << " NEW nodes\n";

        Refine refine(
            *incremental,
            block_points,
            highvoronoi::geometry::KDSearch{8, 1},
            ray_parameters(),
            highvoronoi::SingleThread{},
            highvoronoi::SingleThread{},
            database_parameters(),
            edge_parameters());

        (void)refine.compute();
        replay_points.insert(
            replay_points.end(),
            block_points.begin(),
            block_points.end());

        std::cout << "      appended=" << refine.report().appended_nodes
                  << " NEW vertices=" << refine.report().new_cell_vertices
                  << " affected=" << refine.report().affected_nodes
                  << " invalidated=" << refine.report().invalidated_old_vertices
                  << " repair vertices=" << refine.report().repair_vertices
                  << '\n';

        const bool block_ok = validate_refinement_state(
            *incremental,
            replay_points,
            replay_boundary,
            block + 1);
        all_blocks_ok = all_blocks_ok && block_ok;
    }

    check(all_blocks_ok,
          "the HighVoronoi problem sequence stays consistent and complete in ordinary VoronoiMesh");

    section("Result");
    std::cout << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}


