#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/integration/polygon_integrator.hpp>
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

    Integral integral(mesh, std::size_t{0}, options, 8192, 8192);
    highvoronoi::PolygonAlgorithm<Integral> algorithm;
    const auto report = highvoronoi::integrate(integral, algorithm);

    typename Integral::Data::CellData data;
    double volume_sum = 0.0;
    bool exact = true;
    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!integral.data().read_cell(stable, data)) {
            exact = false;
            continue;
        }

        volume_sum += data.volume();
        exact = exact &&
            std::abs(data.volume() - 1.0 / 16.0) < 2e-9 &&
            data.area().size() == 8;
        for (const double area : data.area()) {
            exact = exact && std::abs(area - 1.0 / 8.0) < 2e-8;
        }
    }

    check(report.updated_cells == 16,
          "4D first polygon pass computes all 16 hypercube cells");
    check(exact,
          "every 4D cell has volume 1/16 and eight facets of measure 1/8");
    check(std::abs(volume_sum - 1.0) < 2e-8,
          "4D polygon volumes sum to unit hypercube volume");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
