#include <highvoronoi/voronoi.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/integration/integrator.hpp>
#include <highvoronoi/integration/polygon_integrator.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;

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

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        65536,
        DatabaseParameters{highvoronoi::DirectHash{65536}});
}

Mesh make_mesh() {
    // Eight generators form a 2x2x2 Cartesian arrangement in the unit cube.
    // Its Voronoi cells have exactly known total volume 1.
    Nodes nodes(Index{8});
    Index next = Index{0};
    for (const Scalar x : {Scalar{0.25}, Scalar{0.75}}) {
        for (const Scalar y : {Scalar{0.25}, Scalar{0.75}}) {
            for (const Scalar z : {Scalar{0.25}, Scalar{0.75}}) {
                nodes.set(next++, point(x, y, z));
            }
        }
    }

    const Boundary boundary = Boundary::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{});

    return Mesh(std::move(nodes), boundary, make_database());
}

void compute_mesh(Mesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    highvoronoi::RaycastParameters<
        highvoronoi::InRangeRaycast,
        Scalar> ray_parameters;
    ray_parameters.variance_tolerance = Scalar{9e-14};

    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters);
    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread>;

    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{});
    compute.compute();
}

highvoronoi::IntegralDataOptions geometry_options() {
    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    return options;
}

bool compare_and_print(
    Mesh& mesh,
    Integral& polygon,
    Integral& fast_polygon) {
    Integral::Data::CellData polygon_cell;
    Integral::Data::CellData fast_cell;

    double volume_sum = 0.0;
    std::size_t interface_entries = 0;
    double max_volume_difference = 0.0;
    double max_area_difference = 0.0;

    for (Index public_cell = Index{0}; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);

        if (!polygon.data().read_cell(stable, polygon_cell) ||
            !fast_polygon.data().read_cell(stable, fast_cell)) {
            return false;
        }
        if (polygon_cell.neighbours() != fast_cell.neighbours() ||
            polygon_cell.area().size() != fast_cell.area().size()) {
            return false;
        }

        volume_sum += fast_cell.volume();
        interface_entries += fast_cell.area().size();
        max_volume_difference = (std::max)(
            max_volume_difference,
            std::abs(polygon_cell.volume() - fast_cell.volume()));

        for (std::size_t i = 0; i < fast_cell.area().size(); ++i) {
            max_area_difference = (std::max)(
                max_area_difference,
                std::abs(polygon_cell.area()[i] - fast_cell.area()[i]));
        }
    }

    std::cout << "FastPolygon volume sum: " << volume_sum << '\n'
              << "Cell-local interface entries: " << interface_entries << '\n'
              << "Max |Polygon-FastPolygon| cell-volume error: "
              << max_volume_difference << '\n'
              << "Max |Polygon-FastPolygon| interface-area error: "
              << max_area_difference << '\n';

    return std::abs(volume_sum - 1.0) < 1e-12 &&
           max_volume_difference < 1e-12 &&
           max_area_difference < 1e-12;
}

} // namespace

int main() {
    Mesh mesh = make_mesh();
    compute_mesh(mesh);

    // Keep independent Integral objects when comparing algorithms: after a
    // successful pass an Integral is clean and a second pass has no work.
    Integral polygon(mesh, 0, geometry_options(), 4096, 4096);
    Integral fast_polygon(mesh, 0, geometry_options(), 4096, 4096);

    highvoronoi::PolygonAlgorithm<Integral> polygon_algorithm;
    auto fast_algorithm =
        highvoronoi::make_fast_polygon_algorithm(fast_polygon);

    const auto polygon_report =
        highvoronoi::integrate(polygon, polygon_algorithm);
    const auto fast_report = highvoronoi::integrate(
        fast_polygon,
        fast_algorithm,
        highvoronoi::ParallelIntegrationExecution{4});

    std::cout << "Polygon updated cells: " << polygon_report.updated_cells << '\n'
              << "Parallel FastPolygon updated cells: "
              << fast_report.updated_cells << '\n';

    const auto cache = fast_algorithm.parallel_cache_stats();
    std::cout << "FastPolygon shared facet cache: hits=" << cache.hits
              << " misses=" << cache.misses
              << " stores=" << cache.stores
              << " entries=" << cache.entries << '\n';

    return compare_and_print(mesh, polygon, fast_polygon) ? 0 : 1;
}
