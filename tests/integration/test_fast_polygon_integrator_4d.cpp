
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
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
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
constexpr int Dimension = 4;

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

Point point(Scalar a, Scalar b, Scalar c, Scalar d) {
    Point result;
    result << a, b, c, d;
    return result;
}

} // namespace

int main() {
    Nodes nodes(Index{16});
    Index cell = 0;
    for (const double a : {0.25, 0.75}) {
        for (const double b : {0.25, 0.75}) {
            for (const double c : {0.25, 0.75}) {
                for (const double d : {0.25, 0.75}) {
                    nodes.set(cell++, point(a, b, c, d));
                }
            }
        }
    }

    const Boundary boundary = Boundary::cuboid(
        point(1.0, 1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0, 0.0),
        std::vector<Index>{});

    auto database = std::make_shared<Database>(
        524288,
        DatabaseParameters{highvoronoi::DirectHash{524288}});
    Mesh mesh(std::move(nodes), boundary, std::move(database));

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
        DatabaseParameters{highvoronoi::DirectHash{524288}},
        EdgeParameters{highvoronoi::DirectHash{524288}});
    compute.compute();

    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = false;
    options.interface_integral = false;

    Integral serial(mesh, std::size_t{0}, options, 8192, 8192);
    Integral parallel(mesh, std::size_t{0}, options, 8192, 8192);
    highvoronoi::FastPolygonAlgorithm<Integral> serial_algorithm;
    highvoronoi::FastPolygonAlgorithm<Integral> parallel_algorithm;
    const auto serial_report = highvoronoi::integrate(serial, serial_algorithm);
    const auto parallel_report = highvoronoi::integrate(
        parallel,
        parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{4});
    const auto stats = parallel_algorithm.parallel_cache_stats();
    std::cout << "parallel cache hits=" << stats.hits
              << " misses=" << stats.misses
              << " stores=" << stats.stores
              << " entries=" << stats.entries << '\n';

    typename Integral::Data::CellData a, b;
    double volume_sum = 0.0;
    bool exact = true;
    bool equal = true;
    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!serial.data().read_cell(stable, a) ||
            !parallel.data().read_cell(stable, b)) {
            exact = false;
            equal = false;
            continue;
        }
        volume_sum += b.volume();
        exact = exact &&
            std::abs(b.volume() - 1.0 / 16.0) < 2e-9 &&
            b.area().size() == 8;
        equal = equal && a.neighbours() == b.neighbours() &&
            std::abs(a.volume() - b.volume()) < 2e-10 &&
            a.area().size() == b.area().size();
        for (std::size_t k = 0; k < b.area().size(); ++k) {
            exact = exact && std::abs(b.area()[k] - 1.0 / 8.0) < 2e-8;
            if (k < a.area().size()) {
                equal = equal &&
                    std::abs(a.area()[k] - b.area()[k]) < 2e-10;
            }
        }
    }

    check(
        serial_report.updated_cells == 16 &&
            parallel_report.updated_cells == 16,
          "4D serial and parallel fast-polygon passes compute all 16 cells");
    check(stats.hits > 0 && stats.misses > 0 && stats.stores > 0,
          "4D parallel FastPolygon recursion exercises the shared facet cache");
    check(equal,
          "4D parallel FastPolygon matches the serial reference cell/interface-wise");
    check(exact,
          "every parallel 4D cell has volume 1/16 and eight facets of measure 1/8");
    check(std::abs(volume_sum - 1.0) < 2e-8,
          "4D parallel fast-polygon volumes sum to unit hypercube volume");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
