#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
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
constexpr Index InitialCount = 16;
constexpr Index AddedCount = 2;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters,
    Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;
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
    std::uniform_real_distribution<Scalar> coordinate(0.04, 0.96);
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
    for (Index i = 0; i < count; ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
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

struct Summary {
    double volume_sum = 0.0;
    bool complete = true;
    bool constant_identities = true;
};

Summary summarize(Mesh& mesh, Integral& integral) {
    Summary result;
    typename Integral::Data::CellData cell;

    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!integral.data().read_cell(stable, cell)) {
            result.complete = false;
            continue;
        }

        result.volume_sum += cell.volume();
        result.constant_identities = result.constant_identities &&
            cell.bulk_integral().size() == 1 &&
            std::abs(cell.bulk_integral()[0] - cell.volume()) < 2e-9 &&
            cell.interface_integral().size() == cell.area().size();

        if (cell.interface_integral().size() == cell.area().size()) {
            for (std::size_t k = 0; k < cell.area().size(); ++k) {
                result.constant_identities = result.constant_identities &&
                    std::abs(cell.interface_integral()[k] - cell.area()[k]) < 2e-9;
            }
        }
    }
    return result;
}

} // namespace

int main() {
    Mesh mesh = make_mesh(InitialCount, 1);
    compute_full_voronoi(mesh);

    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = true;
    options.interface_integral = true;

    Integral integral(mesh, std::size_t{1}, options, 4096, 4096);
    const auto constant = [](const Point&) { return 1.0; };
    auto algorithm = highvoronoi::make_fast_polygon_algorithm(integral, constant);

    const auto initial_report = highvoronoi::integrate(integral, algorithm);
    const Summary initial = summarize(mesh, integral);

    check(initial_report.updated_cells == InitialCount,
          "initial fast-polygon pass computes every NEW cell");
    check(initial.complete && initial.constant_identities,
          "initial f=1 integral satisfies bulk=volume and interface=area");
    check(std::abs(initial.volume_sum - 1.0) < 2e-9,
          "initial fast-polygon volumes sum to one");

    const auto added = random_points(AddedCount, 99);
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

    const auto refine_report = highvoronoi::integrate(integral, algorithm);
    const Summary refined = summarize(mesh, integral);

    std::cout << "refine updated=" << refine_report.updated_cells
              << " of " << mesh.size() << '\n';
    check(mesh.size() == InitialCount + AddedCount,
          "refine appends exactly two cells");
    check(refine_report.updated_cells < static_cast<std::size_t>(mesh.size()),
          "refine leaves CLEAN cells, exercising fast-polygon cleanup reuse");
    check(refined.complete && refined.constant_identities,
          "refine update preserves exact f=1 integral identities");
    check(std::abs(refined.volume_sum - 1.0) < 2e-9,
          "refined fast-polygon volumes still sum to one");

    std::vector<Index> deleted;
    for (Index cell = InitialCount;
         cell < InitialCount + AddedCount;
         ++cell) {
        deleted.push_back(cell);
    }

    Remove remove(
        mesh,
        deleted,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        DatabaseParameters{highvoronoi::DirectHash{262144}},
        EdgeParameters{highvoronoi::DirectHash{262144}});
    (void)remove.remove();
    (void)remove.compute();

    const auto remove_report = highvoronoi::integrate(integral, algorithm);
    const Summary removed = summarize(mesh, integral);

    std::cout << "remove updated=" << remove_report.updated_cells
              << " of " << mesh.size() << '\n';
    check(mesh.size() == InitialCount,
          "remove restores the original public cell count");
    check(remove_report.updated_cells < static_cast<std::size_t>(mesh.size()),
          "remove leaves CLEAN survivors, exercising fast-polygon cleanup reuse");
    check(removed.complete && removed.constant_identities,
          "remove update preserves exact f=1 integral identities");
    check(std::abs(removed.volume_sum - 1.0) < 2e-9,
          "post-remove fast-polygon volumes sum to one");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
