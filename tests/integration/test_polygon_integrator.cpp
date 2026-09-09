#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/integration/polygon_integrator.hpp>
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

constexpr int UnboundedDimension = 2;
using UnboundedDatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using UnboundedDatabase = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock,
    UnboundedDatabaseParameters,
    UnboundedDimension>;
using UnboundedMesh = highvoronoi::VoronoiMesh<
    Scalar,
    UnboundedDimension,
    UnboundedDatabase>;
using UnboundedPoint = UnboundedMesh::NodePoint;
using UnboundedIntegral = highvoronoi::VoronoiIntegral<
    UnboundedMesh,
    double,
    double>;
using UnboundedEdgeParameters = highvoronoi::EdgeBufferParams<>;
using UnboundedRayParameters = highvoronoi::RaycastParameters<
    highvoronoi::ClassicRaycast,
    Scalar>;

UnboundedPoint unbounded_point(Scalar x, Scalar y) {
    UnboundedPoint result;
    result << x, y;
    return result;
}

std::unique_ptr<UnboundedMesh> make_unbounded_cross_mesh() {
    UnboundedMesh::InternalNodes nodes(Index{5});
    nodes.set(Index{0}, unbounded_point(1.0, 0.0));
    nodes.set(Index{1}, unbounded_point(0.0, 1.0));
    nodes.set(Index{2}, unbounded_point(-1.0, 0.0));
    nodes.set(Index{3}, unbounded_point(0.0, -1.0));
    nodes.set(Index{4}, unbounded_point(0.0, 0.0));

    auto database = std::make_shared<UnboundedDatabase>(
        4096,
        UnboundedDatabaseParameters{highvoronoi::DirectHash{4096}});
    auto mesh = std::make_unique<UnboundedMesh>(
        std::move(nodes),
        std::move(database));

    auto tree = highvoronoi::geometry::make_search_tree(
        *mesh,
        highvoronoi::geometry::KDSearch{4, 1});
    auto raycaster = highvoronoi::make_raycaster(
        tree,
        UnboundedRayParameters{});

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        UnboundedMesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        UnboundedDatabaseParameters,
        UnboundedEdgeParameters>;

    Compute compute(
        *mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,
        UnboundedDatabaseParameters{highvoronoi::DirectHash{512}},
        UnboundedEdgeParameters{highvoronoi::DirectHash{512}});
    compute.compute();
    return mesh;
}

struct UnboundedProbeAlgorithm {
    std::vector<Index> seen_cells;

    template <class Update>
    void integrate_cell(Update& update) {
        seen_cells.push_back(static_cast<Index>(update.cell()));
        auto& data = update.data();
        data.set_volume(42.0);

        for (std::size_t ordinal = 0; ordinal < data.area().size(); ++ordinal) {
            data.area()[ordinal] = 10.0 + static_cast<double>(ordinal);
        }
        for (std::size_t component = 0;
             component < data.bulk_integral().size();
             ++component) {
            data.bulk_integral()[component] =
                20.0 + static_cast<double>(component);
        }
        const std::size_t components = data.bulk_integral().size();
        for (std::size_t ordinal = 0;
             ordinal < data.neighbours().size();
             ++ordinal) {
            for (std::size_t component = 0;
                 component < components;
                 ++component) {
                data.interface_integral()[ordinal * components + component] =
                    100.0 + 10.0 * static_cast<double>(ordinal) +
                    static_cast<double>(component);
            }
        }
    }
};

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
    std::cout << "\n[TEST] Polygon constant integration on degenerate 2x2x2 grid\n";

    Mesh mesh = make_mesh(cartesian_2x2x2_points());
    compute_full_voronoi(mesh);

    Integral integral(
        mesh,
        std::size_t{1},
        full_integral_options(),
        4096,
        4096);

    const auto constant = [](const Point&) { return 1.0; };
    auto algorithm = highvoronoi::make_polygon_algorithm(integral, constant);
    const auto report = highvoronoi::integrate(integral, algorithm);

    check(report.updated_cells == 8,
          "first polygon pass computes all eight NEW cells");

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
          "unchanged second polygon pass updates zero cells");
}

void test_degenerate_cartesian_vector_affine() {
    std::cout << "\n[TEST] Polygon vector affine integration on degenerate 2x2x2 grid\n";

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

    auto algorithm = highvoronoi::make_polygon_algorithm(integral, function);
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
    std::cout << "\n[TEST] Polygon geometry on random bounded Voronoi mesh\n";

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
          "random polygon geometry publishes every integral cell");
    check(nonnegative,
          "random polygon geometry produces non-negative volumes and areas");
    check(close(volume_sum, 1.0, 2e-9),
          "random polygon cell volumes sum to the unit-cube volume");
}

void test_unbounded_cells_are_finalized_without_integrator() {
    std::cout << "\n[TEST] unbounded cells bypass integration and publish inf/NaN\n";

    auto mesh_owner = make_unbounded_cross_mesh();
    UnboundedMesh& mesh = *mesh_owner;

    std::size_t infinite_edge_count = 0;
    for (const auto& edge : mesh.infinite_edges()) {
        (void)edge;
        ++infinite_edge_count;
    }
    check(infinite_edge_count == 4,
          "cross mesh exposes four persistent infinite Voronoi edges");

    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = true;
    options.interface_integral = true;

    UnboundedIntegral integral(mesh, std::size_t{2}, options, 1024, 1024);
    UnboundedProbeAlgorithm algorithm;
    const auto report = highvoronoi::integrate(integral, algorithm);

    check(report.updated_cells == 1,
          "only the bounded centre cell enters the concrete integrator pass");
    check(report.unbounded_cells == 4,
          "four unbounded outer cells are finalized outside the integrator");
    check(report.unbounded_interfaces == 8,
          "four unbounded geometric interfaces produce eight cell occurrences");
    check(algorithm.seen_cells.size() == 1,
          "concrete algorithm is never invoked for an unbounded cell");

    constexpr Index centre_public = Index{4};
    const Index centre_stable =
        mesh.index_mapping().public_to_internal(centre_public);
    typename UnboundedIntegral::Data::CellData centre;
    check(integral.data().read_cell(centre_stable, centre),
          "bounded centre integral record is published");
    check(close(centre.volume(), 42.0),
          "bounded centre keeps the value produced by the concrete algorithm");

    bool outer_layout_ok = true;
    bool finite_interfaces_copied = true;
    bool unbounded_interfaces_inf = true;
    bool undefined_function_integrals_nan = true;

    for (Index public_cell = Index{0}; public_cell < Index{4}; ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        typename UnboundedIntegral::Data::CellData outer;
        if (!integral.data().read_cell(stable, outer)) {
            outer_layout_ok = false;
            continue;
        }

        outer_layout_ok = outer_layout_ok &&
            std::isinf(outer.volume()) && outer.volume() > 0.0 &&
            outer.area().size() == outer.neighbours().size() &&
            outer.bulk_integral().size() == 2 &&
            outer.interface_integral().size() == outer.neighbours().size() * 2;

        for (const double value : outer.bulk_integral()) {
            undefined_function_integrals_nan =
                undefined_function_integrals_nan && std::isnan(value);
        }

        std::size_t centre_ordinal = outer.neighbours().size();
        for (std::size_t ordinal = 0;
             ordinal < outer.neighbours().size();
             ++ordinal) {
            if (outer.neighbours()[ordinal] == centre_stable) {
                centre_ordinal = ordinal;
                break;
            }
        }
        if (centre_ordinal == outer.neighbours().size()) {
            finite_interfaces_copied = false;
            continue;
        }

        std::size_t reciprocal_ordinal = centre.neighbours().size();
        for (std::size_t ordinal = 0;
             ordinal < centre.neighbours().size();
             ++ordinal) {
            if (centre.neighbours()[ordinal] == stable) {
                reciprocal_ordinal = ordinal;
                break;
            }
        }
        if (reciprocal_ordinal == centre.neighbours().size()) {
            finite_interfaces_copied = false;
            continue;
        }

        finite_interfaces_copied = finite_interfaces_copied &&
            close(outer.area()[centre_ordinal], centre.area()[reciprocal_ordinal]);
        for (std::size_t component = 0; component < 2; ++component) {
            finite_interfaces_copied = finite_interfaces_copied && close(
                outer.interface_integral()[centre_ordinal * 2 + component],
                centre.interface_integral()[reciprocal_ordinal * 2 + component]);
        }

        for (std::size_t ordinal = 0;
             ordinal < outer.neighbours().size();
             ++ordinal) {
            if (ordinal == centre_ordinal) {
                continue;
            }
            unbounded_interfaces_inf = unbounded_interfaces_inf &&
                std::isinf(outer.area()[ordinal]) &&
                outer.area()[ordinal] > 0.0;
            for (std::size_t component = 0; component < 2; ++component) {
                undefined_function_integrals_nan =
                    undefined_function_integrals_nan &&
                    std::isnan(
                        outer.interface_integral()[ordinal * 2 + component]);
            }
        }
    }

    check(outer_layout_ok,
          "unbounded cells publish +inf volume with complete aligned result layout");
    check(finite_interfaces_copied,
          "finite centre/outer interfaces are copied from the bounded reciprocal cell");
    check(unbounded_interfaces_inf,
          "interfaces containing persistent infinite edges are stored as +infinity");
    check(undefined_function_integrals_nan,
          "not-computed bulk/interface function integrals are stored componentwise as NaN");

    const auto second_report = highvoronoi::integrate(integral, algorithm);
    check(second_report.updated_cells == 0 &&
              second_report.unbounded_cells == 0,
          "unchanged second pass schedules neither finite nor unbounded cells");
}

} // namespace

int main() {
    test_degenerate_cartesian_constant();
    test_degenerate_cartesian_vector_affine();
    test_random_general_position_geometry();
    test_unbounded_cells_are_finalized_without_integrator();

    std::cout << "\nchecks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
