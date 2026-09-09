#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/integration/integration_view.hpp>
#include <highvoronoi/integration/monte_carlo_integrator.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/remove_voronoi.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
constexpr int Dimension = 3;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters,
    Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;
using NeighbourAddress = Mesh::NeighbourAddress;
using Integral = highvoronoi::VoronoiIntegral<Mesh, double, double>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::InRangeRaycast,
    Scalar>;
using Refine = highvoronoi::RefineVoronoi<
    Mesh,
    highvoronoi::geometry::KDSearch,
    RayParameters,
    highvoronoi::SingleThread,
    highvoronoi::SingleThread,
    DatabaseParameters,
    EdgeParameters>;
using Remove = highvoronoi::RemoveVoronoi<
    Mesh,
    highvoronoi::geometry::KDSearch,
    RayParameters,
    highvoronoi::SingleThread,
    highvoronoi::SingleThread,
    DatabaseParameters,
    EdgeParameters>;

std::size_t checks = 0;
std::size_t failures = 0;

void check(bool ok, std::string_view message) {
    ++checks;
    if (ok) {
        std::cout << "[OK]   " << message << '\n';
    } else {
        ++failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

Boundary unit_cube_boundary() {
    return Boundary::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{});
}

std::vector<Point> random_points(Index count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Scalar> coordinate(Scalar{0.02}, Scalar{0.98});
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(count));
    while (points.size() < static_cast<std::size_t>(count)) {
        points.push_back(point(
            coordinate(rng),
            coordinate(rng),
            coordinate(rng)));
    }
    return points;
}

Mesh make_mesh(Index count, std::uint64_t seed) {
    const auto points = random_points(count, seed);
    Nodes nodes(count);
    for (Index i = Index{0}; i < count; ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
    }

    auto database = std::make_shared<Database>(
        262144,
        DatabaseParameters{highvoronoi::DirectHash{262144}});
    return Mesh(std::move(nodes), unit_cube_boundary(), std::move(database));
}

void compute_full_voronoi(Mesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    RayParameters ray_parameters;
    ray_parameters.variance_tolerance = Scalar{9e-14};
    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters);

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
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
        DatabaseParameters{highvoronoi::DirectHash{262144}},
        EdgeParameters{highvoronoi::DirectHash{262144}});
    compute.compute();
}

struct RunResult {
    double volume_sum = 0.0;
    double area_sum = 0.0;
    std::size_t finite_area_entries = 0;
    highvoronoi::IntegrationReport report;
};

RunResult run_case(Index count, std::uint64_t seed) {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] MonteCarloIntegrator on unit cube with " << count << " nodes\n";
    std::cout << "============================================================\n";

    Mesh mesh = make_mesh(count, seed);
    compute_full_voronoi(mesh);

    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = false;
    options.interface_integral = false;

    Integral integral(mesh, std::size_t{0}, options, 4096, 4096);

    highvoronoi::MonteCarloOptions mc_options;
    mc_options.interface_rays = 1000;
    mc_options.bulk_samples_per_ray = 0;
    mc_options.seed = seed ^ 0xa0761d6478bd642fULL;
    mc_options.calculate_area = true;

    highvoronoi::MonteCarloAlgorithm<> algorithm(mc_options);
    RunResult result;
    result.report = highvoronoi::integrate(integral, algorithm);

    typename Integral::Data::CellData cell;
    bool all_cells_complete = true;
    bool all_areas_nonnegative = true;
    for (Index i = Index{0}; i < mesh.size(); ++i) {
        const Index stable = mesh.index_mapping().public_to_internal(i);
        const bool complete = integral.data().read_cell(stable, cell);
        all_cells_complete = all_cells_complete && complete;
        result.volume_sum += cell.volume();
        for (const double area : cell.area()) {
            all_areas_nonnegative = all_areas_nonnegative && area >= 0.0;
            if (area > 0.0) {
                ++result.finite_area_entries;
                result.area_sum += area;
            }
        }
    }

    std::cout << std::setprecision(10);
    std::cout << "nodes:                    " << count << '\n';
    std::cout << "updated cells:            " << result.report.updated_cells << '\n';
    std::cout << "recomputed interfaces:    " << result.report.recomputed_interfaces << '\n';
    std::cout << "sum(volume):              " << result.volume_sum << '\n';
    std::cout << "abs(sum(volume)-1):       " << std::abs(result.volume_sum - 1.0) << '\n';
    std::cout << "sum(area entries):        " << result.area_sum << '\n';
    std::cout << "positive area entries:    " << result.finite_area_entries << '\n';

    check(all_cells_complete,
          "MC integral published a complete snapshot for every cell");
    check(all_areas_nonnegative,
          "all MC area entries are non-negative");
    check(result.report.updated_cells == static_cast<std::size_t>(count),
          "first MC run integrates every new cell");
    check(result.finite_area_entries > 0,
          "MC run produced positive interface/boundary area entries");
    check(std::abs(result.volume_sum - 1.0) < 0.10,
          "sum of MC cell volumes is within 10 percent of the unit cube volume");

    return result;
}

// ============================================================================
// Incremental integral-update regression
// ============================================================================

struct CellSnapshot {
    bool present = false;
    NeighbourAddress neighbour_address = NeighbourAddress{0};
    std::vector<Index> neighbours;
    double volume = 0.0;
    std::vector<double> area;
};

struct IntegralSummary {
    double volume_sum = 0.0;
    double area_sum = 0.0;
    std::size_t positive_area_entries = 0;
    bool all_cells_complete = true;
    bool all_values_finite = true;
    bool all_values_nonnegative = true;
    bool all_neighbour_snapshots_match_mesh = true;
};

highvoronoi::IntegralDataOptions integral_options() {
    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = false;
    options.interface_integral = false;
    return options;
}

highvoronoi::MonteCarloOptions monte_carlo_options(std::uint64_t seed) {
    highvoronoi::MonteCarloOptions options;
    options.interface_rays = 1000;
    options.bulk_samples_per_ray = 0;
    options.seed = seed ^ 0xa0761d6478bd642fULL;
    options.calculate_area = true;
    return options;
}

RayParameters incremental_ray_parameters() {
    RayParameters parameters;
    parameters.variance_tolerance = Scalar{9e-14};
    return parameters;
}

std::vector<CellSnapshot> snapshot_integral(
    const Mesh& mesh,
    const Integral& integral) {
    std::vector<CellSnapshot> result(
        static_cast<std::size_t>(mesh.internal_size()));
    typename Integral::Data::CellData cell;

    for (Index public_cell = Index{0}; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        CellSnapshot& snapshot = result[static_cast<std::size_t>(stable)];
        snapshot.present = integral.data().read_cell(stable, cell);
        if (!snapshot.present) {
            continue;
        }
        snapshot.neighbour_address = cell.neighbour_address();
        snapshot.neighbours.assign(
            cell.neighbours().begin(),
            cell.neighbours().end());
        snapshot.volume = cell.volume();
        snapshot.area.assign(cell.area().begin(), cell.area().end());
    }
    return result;
}

IntegralSummary summarize_integral(
    const Mesh& mesh,
    const Integral& integral) {
    IntegralSummary result;
    typename Integral::Data::CellData cell;
    std::vector<Index> mesh_neighbours;

    for (Index public_cell = Index{0}; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        const bool complete = integral.data().read_cell(stable, cell);
        result.all_cells_complete = result.all_cells_complete && complete;
        if (!complete) {
            continue;
        }

        const bool mesh_neighbours_available =
            mesh.internal_neighbours(stable, mesh_neighbours);
        const bool snapshot_matches =
            mesh_neighbours_available &&
            cell.neighbour_address() == mesh.internal_neighbour_address(stable) &&
            cell.neighbours() == mesh_neighbours;
        result.all_neighbour_snapshots_match_mesh =
            result.all_neighbour_snapshots_match_mesh && snapshot_matches;

        const double volume = cell.volume();
        result.all_values_finite =
            result.all_values_finite && std::isfinite(volume);
        result.all_values_nonnegative =
            result.all_values_nonnegative && volume >= 0.0;
        result.volume_sum += volume;

        result.all_values_nonnegative =
            result.all_values_nonnegative &&
            cell.area().size() == cell.neighbours().size();
        for (const double area : cell.area()) {
            result.all_values_finite =
                result.all_values_finite && std::isfinite(area);
            result.all_values_nonnegative =
                result.all_values_nonnegative && area >= 0.0;
            if (area > 0.0) {
                ++result.positive_area_entries;
                result.area_sum += area;
            }
        }
    }

    return result;
}

std::vector<std::uint8_t> update_mask(
    Integral& integral,
    std::size_t& new_count,
    std::size_t& dirty_old_count,
    std::size_t& clean_old_count) {
    highvoronoi::VoronoiIntegrationView<Integral> view(integral);
    new_count = view.new_count();
    dirty_old_count = view.dirty_old_count();
    clean_old_count = view.clean_old_count();

    std::vector<std::uint8_t> result(
        integral.stable_size(),
        std::uint8_t{0});
    for (std::size_t position = 0; position < view.update_count(); ++position) {
        const Index stable =
            view.stable_internal_index(static_cast<Index>(position));
        result[static_cast<std::size_t>(stable)] = std::uint8_t{1};
    }
    return result;
}

bool clean_snapshots_unchanged(
    const Mesh& mesh,
    const Integral& integral,
    const std::vector<CellSnapshot>& before,
    const std::vector<std::uint8_t>& updated) {
    typename Integral::Data::CellData after;

    for (Index public_cell = Index{0}; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        const std::size_t slot = static_cast<std::size_t>(stable);
        if (slot >= updated.size() || updated[slot] != std::uint8_t{0}) {
            continue;
        }
        if (slot >= before.size() || !before[slot].present) {
            return false;
        }
        if (!integral.data().read_cell(stable, after)) {
            return false;
        }
        if (before[slot].neighbour_address != after.neighbour_address() ||
            before[slot].neighbours != after.neighbours() ||
            before[slot].volume != after.volume() ||
            before[slot].area != after.area()) {
            return false;
        }
    }
    return true;
}

bool current_neighbours_equal_snapshot(
    const Mesh& mesh,
    const std::vector<CellSnapshot>& reference,
    Index stable_limit) {
    std::vector<Index> current;
    for (Index stable = Index{0}; stable < stable_limit; ++stable) {
        const std::size_t slot = static_cast<std::size_t>(stable);
        if (slot >= reference.size() || !reference[slot].present) {
            return false;
        }
        if (!mesh.internal_neighbours(stable, current) ||
            current != reference[slot].neighbours) {
            return false;
        }
    }
    return true;
}

IntegralSummary integrate_fresh_reference(
    Mesh& mesh,
    const highvoronoi::IntegralDataOptions& options,
    const highvoronoi::MonteCarloOptions& mc_options,
    std::size_t expected_cells,
    std::string_view stage) {
    Integral fresh(mesh, std::size_t{0}, options, 4096, 4096);
    highvoronoi::MonteCarloAlgorithm<> algorithm(mc_options);
    const auto report = highvoronoi::integrate(fresh, algorithm);
    const IntegralSummary summary = summarize_integral(mesh, fresh);

    std::cout << "  fresh reference (" << stage << "): updated="
              << report.updated_cells
              << " volume=" << summary.volume_sum << '\n';

    check(report.updated_cells == expected_cells,
          "fresh reference integrates every current cell");
    check(summary.all_cells_complete,
          "fresh reference publishes every current cell");
    check(summary.all_neighbour_snapshots_match_mesh,
          "fresh reference neighbour snapshots match the current mesh exactly");
    check(summary.all_values_finite && summary.all_values_nonnegative,
          "fresh reference contains finite non-negative volume/area data");
    check(std::abs(summary.volume_sum - 1.0) < 0.10,
          "fresh reference volume sum is within 10 percent of the unit cube");

    return summary;
}

void run_incremental_update_case() {
    constexpr Index initial_count = Index{246};
    constexpr Index added_count = Index{14};
    constexpr Index refined_count = initial_count + added_count;
    constexpr std::uint64_t mesh_seed = 0x0ddc0ffeebadf00dULL;
    constexpr std::uint64_t added_seed = 0x6a09e667f3bcc909ULL;

    std::cout << "\n============================================================\n";
    std::cout << "[TEST] MonteCarlo integral update: 246 -> 260 -> 246\n";
    std::cout << "============================================================\n";

    Mesh mesh = make_mesh(initial_count, mesh_seed);
    compute_full_voronoi(mesh);

    const auto options = integral_options();
    const auto mc_options = monte_carlo_options(mesh_seed);
    highvoronoi::MonteCarloAlgorithm<> algorithm(mc_options);
    Integral integral(mesh, std::size_t{0}, options, 4096, 4096);

    const auto initial_report = highvoronoi::integrate(integral, algorithm);
    const IntegralSummary initial_summary = summarize_integral(mesh, integral);
    const std::vector<CellSnapshot> initial_snapshot =
        snapshot_integral(mesh, integral);

    check(initial_report.updated_cells == static_cast<std::size_t>(initial_count),
          "initial 246-cell integral computes every cell");
    check(initial_summary.all_cells_complete,
          "initial 246-cell integral publishes every cell");
    check(initial_summary.all_neighbour_snapshots_match_mesh,
          "initial integral neighbour snapshots match the mesh exactly");
    check(std::abs(initial_summary.volume_sum - 1.0) < 0.10,
          "initial incremental-test volume sum is within 10 percent of 1.0");

    // ---------------------------------------------------------------------
    // Refine: append fourteen NEW nodes, then update only NEW + DIRTY OLD.
    // ---------------------------------------------------------------------
    const std::vector<Point> added_nodes = random_points(added_count, added_seed);
    Refine refine(
        mesh,
        added_nodes,
        highvoronoi::geometry::KDSearch{8, 1},
        incremental_ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        DatabaseParameters{highvoronoi::DirectHash{262144}},
        EdgeParameters{highvoronoi::DirectHash{262144}});
    (void)refine.compute();

    check(mesh.size() == refined_count,
          "refine grows the public mesh from 246 to 260 cells");

    std::size_t refine_new = 0;
    std::size_t refine_dirty = 0;
    std::size_t refine_clean = 0;
    const auto refine_mask = update_mask(
        integral,
        refine_new,
        refine_dirty,
        refine_clean);
    const std::size_t expected_refine_update = refine_new + refine_dirty;

    std::cout << "  refine update prefix: NEW=" << refine_new
              << " DIRTY OLD=" << refine_dirty
              << " CLEAN OLD=" << refine_clean << '\n';

    check(refine_new == static_cast<std::size_t>(added_count),
          "refine integral view classifies exactly fourteen cells as NEW");
    check(refine_dirty > 0,
          "refine marks at least one old integral cell DIRTY");
    check(refine_clean > 0,
          "refine leaves at least one old integral cell CLEAN/reusable");

    const auto refine_report = highvoronoi::integrate(integral, algorithm);
    const IntegralSummary refined_summary = summarize_integral(mesh, integral);

    std::cout << "  refine integral update: updated="
              << refine_report.updated_cells
              << " volume=" << refined_summary.volume_sum << '\n';

    check(refine_report.updated_cells == expected_refine_update,
          "refine integration computes exactly the NEW + DIRTY OLD prefix");
    check(refine_report.updated_cells < static_cast<std::size_t>(refined_count),
          "refine integration reuses CLEAN cells instead of recomputing all 260");
    check(clean_snapshots_unchanged(
              mesh,
              integral,
              initial_snapshot,
              refine_mask),
          "refine leaves every CLEAN old integral record bit-for-bit unchanged");
    check(refined_summary.all_cells_complete,
          "refine update leaves a complete integral snapshot for all 260 cells");
    check(refined_summary.all_neighbour_snapshots_match_mesh,
          "refine update stores exactly the current mesh neighbour snapshot");
    check(refined_summary.all_values_finite &&
              refined_summary.all_values_nonnegative,
          "refine update produces finite non-negative volume/area data");
    check(std::abs(refined_summary.volume_sum - 1.0) < 0.10,
          "refined integral volume sum is within 10 percent of the unit cube");

    {
        const IntegralSummary fresh = integrate_fresh_reference(
            mesh,
            options,
            mc_options,
            static_cast<std::size_t>(refined_count),
            "after refine");
        check(std::abs(refined_summary.volume_sum - fresh.volume_sum) < 0.10,
              "refine-update and fresh full integration agree in total volume within MC tolerance");
    }

    const std::vector<CellSnapshot> refined_snapshot =
        snapshot_integral(mesh, integral);

    // ---------------------------------------------------------------------
    // Remove exactly the fourteen appended public nodes and repair the mesh.
    // The final geometry is the original 246-node generator set again.
    // ---------------------------------------------------------------------
    std::vector<Index> deleted_public_nodes;
    deleted_public_nodes.reserve(static_cast<std::size_t>(added_count));
    for (Index cell = initial_count; cell < refined_count; ++cell) {
        deleted_public_nodes.push_back(cell);
    }

    Remove remove(
        mesh,
        deleted_public_nodes,
        highvoronoi::geometry::KDSearch{8, 1},
        incremental_ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        DatabaseParameters{highvoronoi::DirectHash{262144}},
        EdgeParameters{highvoronoi::DirectHash{262144}});
    (void)remove.remove();
    (void)remove.compute();

    check(mesh.size() == initial_count,
          "remove returns the public mesh from 260 to 246 cells");

    std::size_t remove_new = 0;
    std::size_t remove_dirty = 0;
    std::size_t remove_clean = 0;
    const auto remove_mask = update_mask(
        integral,
        remove_new,
        remove_dirty,
        remove_clean);
    const std::size_t expected_remove_update = remove_new + remove_dirty;

    std::cout << "  remove update prefix: NEW=" << remove_new
              << " DIRTY OLD=" << remove_dirty
              << " CLEAN OLD=" << remove_clean << '\n';

    check(remove_new == 0,
          "remove creates no NEW surviving integral cells");
    check(remove_dirty > 0,
          "remove marks affected surviving integral cells DIRTY");
    check(remove_clean > 0,
          "remove leaves unaffected surviving integral cells CLEAN/reusable");

    const auto remove_report = highvoronoi::integrate(integral, algorithm);
    const IntegralSummary removed_summary = summarize_integral(mesh, integral);

    std::cout << "  remove integral update: updated="
              << remove_report.updated_cells
              << " volume=" << removed_summary.volume_sum << '\n';

    check(remove_report.updated_cells == expected_remove_update,
          "remove integration computes exactly the DIRTY surviving cells");
    check(remove_report.updated_cells < static_cast<std::size_t>(initial_count),
          "remove integration reuses CLEAN cells instead of recomputing all 246");
    check(clean_snapshots_unchanged(
              mesh,
              integral,
              refined_snapshot,
              remove_mask),
          "remove leaves every CLEAN surviving integral record bit-for-bit unchanged");
    check(removed_summary.all_cells_complete,
          "remove update leaves a complete integral snapshot for every survivor");
    check(removed_summary.all_neighbour_snapshots_match_mesh,
          "remove update stores exactly the repaired mesh neighbour snapshot");
    check(removed_summary.all_values_finite &&
              removed_summary.all_values_nonnegative,
          "remove update produces finite non-negative volume/area data");
    check(std::abs(removed_summary.volume_sum - 1.0) < 0.10,
          "post-remove integral volume sum is within 10 percent of the unit cube");

    check(current_neighbours_equal_snapshot(
              mesh,
              initial_snapshot,
              initial_count),
          "removing the fourteen appended nodes restores the original 246-cell neighbour topology");

    {
        const IntegralSummary fresh = integrate_fresh_reference(
            mesh,
            options,
            mc_options,
            static_cast<std::size_t>(initial_count),
            "after remove");
        check(std::abs(removed_summary.volume_sum - fresh.volume_sum) < 0.10,
              "remove-update and fresh full integration agree in total volume within MC tolerance");
    }
}

} // namespace

int main() {
    const RunResult small = run_case(Index{100}, 0x123456789abcdef0ULL);
    const RunResult large = run_case(Index{246}, 0x0ddc0ffeebadf00dULL);

    check(std::abs(large.volume_sum - 1.0) < 0.10,
          "246-node MC volume sum is within 10 percent of 1.0");
    check(std::abs(small.volume_sum - 1.0) < 0.10,
          "100-node MC volume sum is within 10 percent of 1.0");

    run_incremental_update_case();

    std::cout << "\nperformed checks: " << checks << '\n';
    std::cout << "failed checks:    " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
