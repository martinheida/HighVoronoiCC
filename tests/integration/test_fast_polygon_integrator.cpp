#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <array>
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

bool close(double left, double right, double tolerance = 1e-10) {
    return std::abs(left - right) <= tolerance;
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

Mesh make_mesh(const std::vector<Point>& points) {
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

highvoronoi::IntegralDataOptions full_integral_options() {
    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = true;
    options.interface_integral = true;
    return options;
}

std::vector<Point> cartesian_2x2x2_points() {
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

void test_degenerate_cartesian_constant() {
    std::cout << "\n[TEST] FastPolygon constant integration on degenerate 2x2x2 grid\n";

    Mesh mesh = make_mesh(cartesian_2x2x2_points());
    compute_full_voronoi(mesh);

    Integral integral(
        mesh,
        std::size_t{1},
        full_integral_options(),
        4096,
        4096);

    const auto constant = [](const Point&) { return 1.0; };
    auto algorithm = highvoronoi::make_fast_polygon_algorithm(integral, constant);
    const auto report = highvoronoi::integrate(integral, algorithm);

    check(report.updated_cells == 8,
          "first fast-polygon pass computes all eight NEW cells");

    typename Integral::Data::CellData cell;
    double volume_sum = 0.0;
    bool exact_cells = true;
    bool exact_interfaces = true;

    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!integral.data().read_cell(stable, cell)) {
            exact_cells = false;
            continue;
        }

        volume_sum += cell.volume();
        exact_cells = exact_cells &&
            close(cell.volume(), 0.125) &&
            cell.neighbours().size() == 6 &&
            cell.area().size() == 6 &&
            cell.bulk_integral().size() == 1 &&
            close(cell.bulk_integral()[0], 0.125) &&
            cell.interface_integral().size() == 6;

        for (std::size_t k = 0; k < cell.area().size(); ++k) {
            exact_interfaces = exact_interfaces &&
                close(cell.area()[k], 0.25) &&
                close(cell.interface_integral()[k], 0.25);
        }
    }

    check(exact_cells,
          "every Cartesian cell has V=1/8, six facets and bulk(f=1)=V");
    check(exact_interfaces,
          "every Cartesian facet has A=1/4 and interface integral(f=1)=A");
    check(close(volume_sum, 1.0),
          "degenerate Cartesian cell volumes sum exactly to unit-cube volume");

    const auto second_report = highvoronoi::integrate(integral, algorithm);
    check(second_report.updated_cells == 0,
          "unchanged second fast-polygon pass updates zero cells");
}

void test_degenerate_cartesian_vector_affine() {
    std::cout << "\n[TEST] FastPolygon vector affine integration on degenerate 2x2x2 grid\n";

    Mesh mesh = make_mesh(cartesian_2x2x2_points());
    compute_full_voronoi(mesh);

    Integral integral(
        mesh,
        std::size_t{2},
        full_integral_options(),
        4096,
        4096);

    const auto function = [](const Point& x) {
        return std::array<double, 2>{
            1.0,
            1.0 + 2.0 * x[0] + 3.0 * x[1] + 4.0 * x[2]};
    };

    auto algorithm = highvoronoi::make_fast_polygon_algorithm(integral, function);
    (void)highvoronoi::integrate(integral, algorithm);

    typename Integral::Data::CellData cell;
    double volume_sum = 0.0;
    double constant_bulk_sum = 0.0;
    double affine_bulk_sum = 0.0;
    bool layout_ok = true;

    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!integral.data().read_cell(stable, cell)) {
            layout_ok = false;
            continue;
        }

        volume_sum += cell.volume();
        layout_ok = layout_ok &&
            cell.bulk_integral().size() == 2 &&
            cell.interface_integral().size() ==
                cell.neighbours().size() * 2;
        if (!layout_ok) {
            continue;
        }

        constant_bulk_sum += cell.bulk_integral()[0];
        affine_bulk_sum += cell.bulk_integral()[1];
        layout_ok = layout_ok && close(cell.bulk_integral()[0], cell.volume());

        for (std::size_t ordinal = 0;
             ordinal < cell.neighbours().size();
             ++ordinal) {
            layout_ok = layout_ok && close(
                cell.interface_integral()[ordinal * 2],
                cell.area()[ordinal]);
        }
    }

    check(layout_ok,
          "vector integrals retain neighbour-major two-component layout");
    check(close(volume_sum, 1.0) && close(constant_bulk_sum, 1.0),
          "first vector component f=1 integrates exactly to unit volume");
    check(close(affine_bulk_sum, 5.5),
          "affine component integrates exactly over the unit cube");
}

void test_random_general_position_geometry() {
    std::cout << "\n[TEST] FastPolygon geometry on random bounded Voronoi mesh\n";

    std::mt19937_64 rng(0x504f4c59474f4eULL);
    std::uniform_real_distribution<double> coordinate(0.05, 0.95);
    std::vector<Point> points;
    points.reserve(12);
    for (std::size_t i = 0; i < 12; ++i) {
        points.push_back(point(
            coordinate(rng),
            coordinate(rng),
            coordinate(rng)));
    }

    Mesh mesh = make_mesh(points);
    compute_full_voronoi(mesh);

    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = false;
    options.interface_integral = false;

    Integral integral(mesh, std::size_t{0}, options, 4096, 4096);
    highvoronoi::PolygonAlgorithm<Integral> algorithm;
    (void)highvoronoi::integrate(integral, algorithm);

    typename Integral::Data::CellData cell;
    double volume_sum = 0.0;
    bool complete = true;
    bool nonnegative = true;
    for (Index public_cell = 0; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        complete = complete && integral.data().read_cell(stable, cell);
        volume_sum += cell.volume();
        nonnegative = nonnegative && cell.volume() >= 0.0;
        for (const double area : cell.area()) {
            nonnegative = nonnegative && area >= 0.0;
        }
    }

    check(complete,
          "random fast-polygon geometry publishes every integral cell");
    check(nonnegative,
          "random fast-polygon geometry produces non-negative volumes and areas");
    check(close(volume_sum, 1.0, 2e-9),
          "random fast-polygon cell volumes sum to the unit-cube volume");
}

} // namespace

int main() {
    test_degenerate_cartesian_constant();
    test_degenerate_cartesian_vector_affine();
    test_random_general_position_geometry();

    std::cout << "\nchecks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
