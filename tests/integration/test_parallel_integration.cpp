
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/integration/heuristic_integrator.hpp>
#include <highvoronoi/integration/heuristic_mc_integrator.hpp>
#include <highvoronoi/integration/integration_view.hpp>
#include <highvoronoi/integration/monte_carlo_integrator.hpp>
#include <highvoronoi/integration/polygon_integrator.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
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

/** Intentionally hides make_worker_algorithm() from one source algorithm. */
template <class Algorithm>
class SerialOnlyAlgorithm final {
public:
    explicit SerialOnlyAlgorithm(Algorithm& algorithm)
        : algorithm_(&algorithm) {}

    template <class Pass>
    void begin_pass(Pass& pass) {
        algorithm_->begin_pass(pass);
    }

    template <class Pass>
    void integrate_cell(Pass& pass, std::size_t position) {
        algorithm_->integrate_cell(pass, position);
    }

    template <class Pass>
    void cleanup_cell(Pass& pass, std::size_t position) {
        algorithm_->cleanup_cell(pass, position);
    }

    template <class Pass>
    void end_pass(Pass& pass) {
        algorithm_->end_pass(pass);
    }

private:
    Algorithm* algorithm_ = nullptr;
};

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

bool close(double left, double right, double tolerance = 1e-10) {
    return std::abs(left - right) <= tolerance;
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

Mesh::BoundaryType unit_cube_boundary() {
    return Mesh::BoundaryType::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{});
}

std::vector<Point> cartesian_points() {
    std::vector<Point> points;
    points.reserve(8);
    for (const double x : {0.25, 0.75}) {
        for (const double y : {0.25, 0.75}) {
            for (const double z : {0.25, 0.75}) {
                points.push_back(point(x, y, z));
            }
        }
    }
    return points;
}

Mesh make_mesh() {
    const auto points = cartesian_points();
    Nodes nodes(static_cast<Index>(points.size()));
    for (std::size_t i = 0; i < points.size(); ++i) {
        nodes.set(static_cast<Index>(i), points[i]);
    }

    auto database = std::make_shared<Database>(
        262144,
        DatabaseParameters{highvoronoi::DirectHash{262144}});
    return Mesh(
        std::move(nodes),
        unit_cube_boundary(),
        std::move(database));
}

RayParameters ray_parameters() {
    RayParameters parameters;
    parameters.variance_tolerance = Scalar{9e-14};
    return parameters;
}

void compute_full_voronoi(Mesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});
    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters());

    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        decltype(raycaster),
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

highvoronoi::IntegralDataOptions full_integral_options() {
    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = true;
    options.interface_integral = true;
    return options;
}

highvoronoi::IntegralDataOptions geometry_integral_options() {
    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = false;
    options.interface_integral = false;
    return options;
}

bool equal_integrals(
    Mesh& mesh,
    Integral& first,
    Integral& second,
    double tolerance = 2e-10) {
    typename Integral::Data::CellData a;
    typename Integral::Data::CellData b;

    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!first.data().read_cell(stable, a) ||
            !second.data().read_cell(stable, b)) {
            return false;
        }
        if (a.neighbours() != b.neighbours() ||
            !close(a.volume(), b.volume(), tolerance) ||
            a.area().size() != b.area().size() ||
            a.bulk_integral().size() != b.bulk_integral().size() ||
            a.interface_integral().size() != b.interface_integral().size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.area().size(); ++i) {
            if (!close(a.area()[i], b.area()[i], tolerance)) {
                return false;
            }
        }
        for (std::size_t i = 0; i < a.bulk_integral().size(); ++i) {
            if (!close(a.bulk_integral()[i], b.bulk_integral()[i], tolerance)) {
                return false;
            }
        }
        for (std::size_t i = 0; i < a.interface_integral().size(); ++i) {
            if (!close(a.interface_integral()[i], b.interface_integral()[i], tolerance)) {
                return false;
            }
        }
    }
    return true;
}

void test_worker_neighbour_presentation_permutation() {
    std::cout << "\n[TEST] Worker-view neighbour/payload permutation\n";

    Mesh mesh = make_mesh();
    compute_full_voronoi(mesh);

    Integral integral(mesh, 2, full_integral_options(), 4096, 4096);
    const auto function = [](const Point& x) {
        return std::array<double, 2>{
            1.0,
            1.0 + 2.0 * x[0] + 3.0 * x[1] + 4.0 * x[2]};
    };
    auto algorithm = highvoronoi::make_polygon_algorithm(integral, function);
    (void)highvoronoi::integrate(integral, algorithm);

    // Make two old cells DIRTY without changing mesh topology. The master view
    // therefore starts [stable 0, stable 7, CLEAN...]. Giving only master cell
    // 1 to the worker creates worker public order [stable 7, stable 0, ...].
    integral.dirty_tracker()->set_dirty(std::size_t{0}, true);
    integral.dirty_tracker()->set_dirty(std::size_t{7}, true);

    auto master = highvoronoi::make_integration_view(integral);
    check(master.update_count() == 2,
          "master view contains the two explicitly dirty cells");
    auto worker = master.make_worker_view(1, 2);
    check(worker->stable_internal_index(Index{0}) == Index{7} &&
              worker->stable_internal_index(Index{1}) == Index{0},
          "worker moves its assigned master range to the public prefix");

    typename Integral::Data::CellData raw_before;
    const Index stable_cell = worker->stable_internal_index(Index{0});
    const bool old_complete = integral.data().read_cell(stable_cell, raw_before);
    check(old_complete,
          "raw committed cell exists before worker-view preparation");

    typename decltype(master)::CellUpdate update;
    worker->prepare_update(Index{0}, update);

    bool mapping_ok = true;
    bool payload_ok = true;
    bool saw_changed_public_index = false;
    const auto& raw_neighbours = update.new_internal_neighbours();
    const auto& presented = update.neighbours();

    for (std::size_t k = 0; k < presented.size(); ++k) {
        const std::size_t raw_ordinal = update.raw_neighbour_ordinal(k);
        if (raw_ordinal >= raw_neighbours.size()) {
            mapping_ok = false;
            continue;
        }

        const Index raw_neighbour = raw_neighbours[raw_ordinal];
        Index expected_public{};
        if (static_cast<std::size_t>(raw_neighbour) <
            static_cast<std::size_t>(mesh.internal_size())) {
            const auto wrapped_public =
                mesh.index_mapping().internal_to_public(raw_neighbour);
            if (!wrapped_public) {
                mapping_ok = false;
                continue;
            }
            const auto worker_public =
                worker->mesh().reordered_public_index(*wrapped_public);
            if (!worker_public) {
                mapping_ok = false;
                continue;
            }
            expected_public = *worker_public;
            saw_changed_public_index = saw_changed_public_index ||
                expected_public != raw_neighbour;
        } else {
            const Index invalid = (std::numeric_limits<Index>::max)();
            if (raw_neighbour == invalid) {
                mapping_ok = false;
                continue;
            }
            const Index plane = invalid - Index{1} - raw_neighbour;
            expected_public = static_cast<Index>(
                worker->size() + static_cast<std::size_t>(plane));
        }

        mapping_ok = mapping_ok && presented[k] == expected_public;

        // If this interface survives the dirty invalidation, the presented area
        // must be exactly the payload belonging to the same raw ordinal.
        if (!update.interface_requires_recompute(k)) {
            payload_ok = payload_ok &&
                raw_ordinal < raw_before.area().size() &&
                close(update.data().area()[k], raw_before.area()[raw_ordinal]);
        }
    }

    check(std::is_sorted(presented.begin(), presented.end()),
          "worker neighbours are sorted in worker-public numbering");
    check(saw_changed_public_index,
          "worker public neighbour indices genuinely differ from stable-internal indices");
    check(mapping_ok,
          "every worker neighbour maps through its preserved raw ordinal");
    check(payload_ok,
          "surviving area payloads follow the identical raw-ordinal permutation");
}

void test_serial_progress_overload() {
    std::cout << "\n[TEST] Serial integration progress\n";

    Mesh mesh = make_mesh();
    compute_full_voronoi(mesh);
    Integral integral(mesh, 0, geometry_integral_options(), 4096, 4096);
    auto algorithm = highvoronoi::make_polygon_algorithm(integral);

    std::ostringstream output;
    highvoronoi::IntegrationProgressOptions progress;
    progress.enabled = true;
    progress.delta_t = std::chrono::hours(24);
    progress.subject = "serial-integral";
    progress.output = &output;

    const auto report = highvoronoi::integrate(integral, algorithm, progress);
    check(report.updated_cells == 8,
          "serial progress overload computes every NEW cell");
    check(output.str().find("serial-integral 100.0 %") != std::string::npos,
          "serial progress meter reaches 100 percent");
}

void test_polygon_parallel_and_incremental() {
    std::cout << "\n[TEST] Polygon serial/parallel equivalence\n";

    Mesh mesh = make_mesh();
    compute_full_voronoi(mesh);

    Integral serial(mesh, 2, full_integral_options(), 4096, 4096);
    Integral parallel(mesh, 2, full_integral_options(), 4096, 4096);

    const auto function = [](const Point& x) {
        return std::array<double, 2>{
            1.0,
            1.0 + 2.0 * x[0] + 3.0 * x[1] + 4.0 * x[2]};
    };

    auto serial_algorithm =
        highvoronoi::make_polygon_algorithm(serial, function);
    auto parallel_algorithm =
        highvoronoi::make_polygon_algorithm(parallel, function);

    const auto serial_report =
        highvoronoi::integrate(serial, serial_algorithm);

    std::ostringstream progress_output;
    highvoronoi::IntegrationProgressOptions progress;
    progress.enabled = true;
    progress.delta_t = std::chrono::hours(24);
    progress.subject = "parallel-integral";
    progress.output = &progress_output;

    // Eight cells over three workers deliberately exercises non-divisible
    // contiguous ranges [0,2), [2,5), [5,8).
    const auto parallel_report = highvoronoi::integrate(
        parallel,
        parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{3},
        progress);

    check(serial_report.updated_cells == 8 &&
              parallel_report.updated_cells == 8,
          "serial and parallel Polygon compute all NEW cells");
    check(equal_integrals(mesh, serial, parallel),
          "parallel Polygon is numerically identical to serial Polygon");
    check(progress_output.str().find("parallel-integral 100.0 %") !=
              std::string::npos,
          "parallel progress meter reaches exactly 100 percent");

    std::vector<Point> added{
        point(0.43, 0.57, 0.36),
        point(0.62, 0.41, 0.68)};
    Refine refine(
        mesh,
        added,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        DatabaseParameters{highvoronoi::DirectHash{262144}},
        EdgeParameters{highvoronoi::DirectHash{262144}});
    (void)refine.compute();

    const auto serial_refined =
        highvoronoi::integrate(serial, serial_algorithm);
    const auto parallel_refined = highvoronoi::integrate(
        parallel,
        parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    check(serial_refined.updated_cells > 0 &&
              serial_refined.updated_cells == parallel_refined.updated_cells,
          "serial and parallel Polygon update the same refine work set");
    check(equal_integrals(mesh, serial, parallel, 5e-10),
          "parallel Polygon remains identical after refine with CLEAN context");
}

void test_monte_carlo_parallel_cleanup() {
    std::cout << "\n[TEST] Monte-Carlo parallel first pass + serial cleanup\n";

    Mesh mesh = make_mesh();
    compute_full_voronoi(mesh);

    Integral integral(mesh, 0, geometry_integral_options(), 4096, 4096);
    highvoronoi::MonteCarloOptions options;
    options.interface_rays = 20000;
    options.seed = 123;
    highvoronoi::MonteCarloAlgorithm<> algorithm(options);

    const auto report = highvoronoi::integrate(
        integral,
        algorithm,
        highvoronoi::ParallelIntegrationExecution{4});

    typename Integral::Data::CellData cell;
    typename Integral::Data::CellData opposite;
    double volume_sum = 0.0;
    bool reciprocal_areas_equal = true;

    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!integral.data().read_cell(stable, cell)) {
            reciprocal_areas_equal = false;
            continue;
        }
        volume_sum += cell.volume();

        for (std::size_t ordinal = 0;
             ordinal < cell.neighbours().size();
             ++ordinal) {
            const Index neighbour = cell.neighbours()[ordinal];
            if (static_cast<std::size_t>(neighbour) >=
                static_cast<std::size_t>(mesh.internal_size())) {
                continue; // boundary mirror
            }
            if (!integral.data().read_cell(neighbour, opposite)) {
                reciprocal_areas_equal = false;
                continue;
            }
            const auto found = std::find(
                opposite.neighbours().begin(),
                opposite.neighbours().end(),
                stable);
            if (found == opposite.neighbours().end()) {
                reciprocal_areas_equal = false;
                continue;
            }
            const std::size_t other_ordinal = static_cast<std::size_t>(
                std::distance(opposite.neighbours().begin(), found));
            reciprocal_areas_equal = reciprocal_areas_equal &&
                close(cell.area()[ordinal],
                      opposite.area()[other_ordinal],
                      1e-12);
        }
    }

    check(report.updated_cells == 8,
          "parallel Monte-Carlo computes all NEW cells");
    check(reciprocal_areas_equal,
          "serial cleanup symmetrizes Monte-Carlo areas across worker ranges");
    check(std::abs(volume_sum - 1.0) < 0.04,
          "parallel Monte-Carlo volume remains close to unit-cube volume");

    const auto unchanged = highvoronoi::integrate(
        integral,
        algorithm,
        highvoronoi::ParallelIntegrationExecution{4});
    check(unchanged.updated_cells == 0,
          "unchanged parallel Monte-Carlo pass updates zero cells");
}

void test_fast_polygon_parallel_and_incremental() {
    std::cout << "\n[TEST] FastPolygon serial/parallel shared-cache equivalence\n";

    Mesh mesh = make_mesh();
    compute_full_voronoi(mesh);

    Integral serial(mesh, 2, full_integral_options(), 4096, 4096);
    Integral parallel(mesh, 2, full_integral_options(), 4096, 4096);

    const auto function = [](const Point& x) {
        return std::array<double, 2>{
            1.0,
            1.0 + 2.0 * x[0] + 3.0 * x[1] + 4.0 * x[2]};
    };

    auto serial_algorithm =
        highvoronoi::make_fast_polygon_algorithm(serial, function);
    auto parallel_algorithm =
        highvoronoi::make_fast_polygon_algorithm(parallel, function);

    const auto serial_report =
        highvoronoi::integrate(serial, serial_algorithm);
    const auto serial_parallel_stats =
        serial_algorithm.parallel_cache_stats();
    check(
        serial_parallel_stats.hits == 0 &&
            serial_parallel_stats.misses == 0 &&
            serial_parallel_stats.stores == 0 &&
            serial_parallel_stats.entries == 0,
        "serial FastPolygon does not initialize the shared parallel cache");

    const auto parallel_report = highvoronoi::integrate(
        parallel,
        parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    check(serial_report.updated_cells == 8 &&
              parallel_report.updated_cells == 8,
          "serial and parallel FastPolygon compute all NEW cells");
    check(equal_integrals(mesh, serial, parallel, 5e-10),
          "parallel FastPolygon is numerically identical to serial FastPolygon");

    const auto first_stats = parallel_algorithm.parallel_cache_stats();
    check(first_stats.hits > 0 &&
              first_stats.misses > 0 &&
              first_stats.stores > 0 &&
              first_stats.entries > 0,
          "parallel FastPolygon actually uses the shared recursive facet cache");

    std::vector<Point> added{
        point(0.43, 0.57, 0.36),
        point(0.62, 0.41, 0.68)};
    Refine refine(
        mesh,
        added,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        DatabaseParameters{highvoronoi::DirectHash{262144}},
        EdgeParameters{highvoronoi::DirectHash{262144}});
    (void)refine.compute();

    const auto serial_refined =
        highvoronoi::integrate(serial, serial_algorithm);
    const auto parallel_refined = highvoronoi::integrate(
        parallel,
        parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    check(serial_refined.updated_cells > 0 &&
              serial_refined.updated_cells == parallel_refined.updated_cells,
          "serial and parallel FastPolygon update the same refine work set");
    check(equal_integrals(mesh, serial, parallel, 1e-9),
          "parallel FastPolygon remains identical after refine with CLEAN context");

    const auto refined_stats = parallel_algorithm.parallel_cache_stats();
    check(refined_stats.entries > 0 && refined_stats.stores > 0,
          "refined parallel pass rebuilds a fresh shared facet hierarchy");

    const auto unchanged = highvoronoi::integrate(
        parallel,
        parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});
    check(unchanged.updated_cells == 0,
          "unchanged parallel FastPolygon pass updates zero cells");
}


void test_heuristic_parallel_and_forced_source_update() {
    std::cout << "\n[TEST] Heuristic serial/parallel + forced source update\n";

    Mesh mesh = make_mesh();
    compute_full_voronoi(mesh);

    Integral source(mesh, 0, geometry_integral_options(), 4096, 4096);
    auto source_algorithm =
        highvoronoi::make_fast_polygon_algorithm(source);
    (void)highvoronoi::integrate(source, source_algorithm);

    Integral serial(mesh, 2, full_integral_options(), 4096, 4096);
    Integral parallel(mesh, 2, full_integral_options(), 4096, 4096);
    const auto function = [](const Point& x) {
        return std::array<double, 2>{
            1.0,
            1.0 + 2.0 * x[0] + 3.0 * x[1] + 4.0 * x[2]};
    };

    auto serial_algorithm = highvoronoi::make_heuristic_algorithm(
        serial,
        source,
        source_algorithm,
        function);
    auto parallel_algorithm = highvoronoi::make_heuristic_algorithm(
        parallel,
        source,
        source_algorithm,
        function);

    const auto serial_report =
        highvoronoi::integrate(serial, serial_algorithm);
    const auto parallel_report = highvoronoi::integrate(
        parallel,
        parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    check(serial_report.updated_cells == 8 &&
              parallel_report.updated_cells == 8,
          "serial and parallel Heuristic compute all NEW target cells");
    check(equal_integrals(mesh, serial, parallel, 2e-12),
          "parallel Heuristic is cell/interface identical to serial Heuristic");

    // A fresh source exercises execution-aware force_source_update. The outer
    // target requests three workers; FastPolygon is parallel-capable, so the
    // nested source update must use its shared parallel facet cache as well.
    Integral forced_source(
        mesh,
        0,
        geometry_integral_options(),
        4096,
        4096);
    auto forced_source_algorithm =
        highvoronoi::make_fast_polygon_algorithm(forced_source);
    Integral forced_target(
        mesh,
        1,
        full_integral_options(),
        4096,
        4096);
    const auto one = [](const Point&) { return 1.0; };
    auto forced_algorithm = highvoronoi::make_heuristic_algorithm(
        forced_target,
        forced_source,
        forced_source_algorithm,
        one,
        highvoronoi::HeuristicOptions{true});

    const auto forced_report = highvoronoi::integrate(
        forced_target,
        forced_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});
    const auto source_stats =
        forced_source_algorithm.parallel_cache_stats();

    check(forced_report.updated_cells == 8,
          "parallel Heuristic force-updates missing source geometry and target");
    check(source_stats.entries > 0 && source_stats.stores > 0,
          "force_source_update inherits parallel execution for capable source algorithm");

    typename Integral::Data::CellData source_cell;
    typename Integral::Data::CellData target_cell;
    bool constant_geometry_identity = true;
    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        constant_geometry_identity = constant_geometry_identity &&
            forced_source.data().read_geometry_cell(stable, source_cell) &&
            forced_target.data().read_cell(stable, target_cell);
        if (!constant_geometry_identity) {
            break;
        }
        constant_geometry_identity = constant_geometry_identity &&
            close(source_cell.volume(), target_cell.volume(), 1e-12) &&
            source_cell.area().size() == target_cell.area().size() &&
            target_cell.interface_integral().size() == target_cell.area().size();
        for (std::size_t ordinal = 0;
             constant_geometry_identity && ordinal < source_cell.area().size();
             ++ordinal) {
            constant_geometry_identity = constant_geometry_identity &&
                close(source_cell.area()[ordinal],
                      target_cell.area()[ordinal],
                      1e-12) &&
                close(target_cell.interface_integral()[ordinal],
                      source_cell.area()[ordinal],
                      1e-12);
        }
    }
    check(constant_geometry_identity,
          "parallel forced Heuristic f=1 preserves source volume/area and interface identity");

    Integral serial_only_source(
        mesh,
        0,
        geometry_integral_options(),
        4096,
        4096);
    auto serial_only_base =
        highvoronoi::make_fast_polygon_algorithm(serial_only_source);
    SerialOnlyAlgorithm<decltype(serial_only_base)> serial_only_algorithm(
        serial_only_base);
    Integral serial_only_target(
        mesh,
        1,
        full_integral_options(),
        4096,
        4096);
    auto serial_fallback = highvoronoi::make_heuristic_algorithm(
        serial_only_target,
        serial_only_source,
        serial_only_algorithm,
        one,
        highvoronoi::HeuristicOptions{true});
    const auto serial_fallback_report = highvoronoi::integrate(
        serial_only_target,
        serial_fallback,
        highvoronoi::ParallelIntegrationExecution{3});
    const auto serial_source_parallel_stats =
        serial_only_base.parallel_cache_stats();

    check(serial_fallback_report.updated_cells == 8,
          "parallel Heuristic accepts an intentionally serial-only source provider");
    check(serial_source_parallel_stats.entries == 0 &&
              serial_only_base.cache_stats().entries > 0,
          "serial-only forced source falls back to serial update while target remains parallel");
}

void test_heuristic_mc_parallel_staged_reference() {
    std::cout << "\n[TEST] HeuristicMC parallel staged-reference equivalence\n";

    Mesh mesh = make_mesh();
    compute_full_voronoi(mesh);

    highvoronoi::MonteCarloOptions options;
    options.interface_rays = 20000;
    options.seed = 0x12345678ULL;
    options.calculate_area = true;
    options.heuristic = true;
    options.bulk_samples_per_ray = 1;

    const auto function = [](const Point& x) {
        return std::array<double, 2>{
            1.0,
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2]};
    };

    // Staged reference: identical parallel MC geometry first, then an ordinary
    // parallel Heuristic pass over the finalized/symmetrized source areas.
    Integral geometry(mesh, 0, geometry_integral_options(), 4096, 4096);
    highvoronoi::MonteCarloAlgorithm<> geometry_algorithm(options);
    const auto geometry_report = highvoronoi::integrate(
        geometry,
        geometry_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    Integral staged(mesh, 2, full_integral_options(), 4096, 4096);
    auto staged_algorithm = highvoronoi::make_heuristic_algorithm(
        staged,
        geometry,
        geometry_algorithm,
        function);
    const auto staged_report = highvoronoi::integrate(
        staged,
        staged_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    Integral combined(mesh, 2, full_integral_options(), 4096, 4096);
    auto combined_algorithm = highvoronoi::make_heuristic_mc_algorithm(
        combined,
        function,
        options);
    const auto combined_report = highvoronoi::integrate(
        combined,
        combined_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    check(geometry_report.updated_cells == 8 &&
              staged_report.updated_cells == 8 &&
              combined_report.updated_cells == 8,
          "parallel staged and combined HeuristicMC compute all NEW cells");
    check(equal_integrals(mesh, staged, combined, 2e-12),
          "parallel HeuristicMC equals parallel MC-cleanup followed by parallel Heuristic");

    typename Integral::Data::CellData cell;
    bool interface_uses_final_area = true;
    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!combined.data().read_cell(stable, cell)) {
            interface_uses_final_area = false;
            continue;
        }
        for (std::size_t ordinal = 0;
             ordinal < cell.area().size();
             ++ordinal) {
            interface_uses_final_area = interface_uses_final_area &&
                close(cell.interface_integral()[ordinal * 2],
                      cell.area()[ordinal],
                      1e-12);
        }
    }
    check(interface_uses_final_area,
          "HeuristicMC f=1 interface component is built from finalized MC areas");

    std::vector<Point> added{
        point(0.43, 0.57, 0.36),
        point(0.62, 0.41, 0.68)};
    Refine refine(
        mesh,
        added,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        DatabaseParameters{highvoronoi::DirectHash{262144}},
        EdgeParameters{highvoronoi::DirectHash{262144}});
    (void)refine.compute();

    const auto geometry_refined = highvoronoi::integrate(
        geometry,
        geometry_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});
    const auto staged_refined = highvoronoi::integrate(
        staged,
        staged_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});
    const auto combined_refined = highvoronoi::integrate(
        combined,
        combined_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    check(geometry_refined.updated_cells > 0 &&
              geometry_refined.updated_cells == staged_refined.updated_cells &&
              staged_refined.updated_cells == combined_refined.updated_cells,
          "parallel HeuristicMC staged/combined update the same refine work set");
    check(equal_integrals(mesh, staged, combined, 2e-12),
          "parallel HeuristicMC remains staged-reference identical after refine/CLEAN reuse");

    const auto unchanged = highvoronoi::integrate(
        combined,
        combined_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});
    check(unchanged.updated_cells == 0,
          "unchanged parallel HeuristicMC pass updates zero cells");
}

} // namespace

int main() {
    test_worker_neighbour_presentation_permutation();
    test_serial_progress_overload();
    test_polygon_parallel_and_incremental();
    test_monte_carlo_parallel_cleanup();
    test_fast_polygon_parallel_and_incremental();
    test_heuristic_parallel_and_forced_source_update();
    test_heuristic_mc_parallel_staged_reference();

    std::cout << "\nchecks=" << checks
              << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
