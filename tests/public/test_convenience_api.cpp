#include <highvoronoi/voronoi.hpp>
#include <highvoronoi/high_voronoi.hpp>
#include <highvoronoi/integrals.hpp>

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {

constexpr int Dimension = 2;
using UserBoundary = highvoronoi::Boundary<Dimension>;
using UserPoint = UserBoundary::Point;

UserBoundary unit_square() {
    return UserBoundary::cuboid(
        UserPoint::Constant(1.0),
        UserPoint::Zero(),
        std::vector<std::size_t>{});
}

constexpr double initial_points[] = {
    0.15, 0.20,
    0.78, 0.18,
    0.22, 0.74,
    0.76, 0.79,
    0.48, 0.43,
    0.51, 0.88,
    0.89, 0.51,
    0.09, 0.49
};

constexpr double added_points[] = {
    0.34, 0.55,
    0.64, 0.60
};

} // namespace

int main() {
    using DefaultParams = highvoronoi::DataBaseParams<>;
    using DefaultRay = highvoronoi::RaycastParameters<>;

    static_assert(std::is_same_v<typename DefaultParams::Scalar, double>);
    static_assert(std::is_same_v<typename DefaultParams::Index, std::uint32_t>);
    static_assert(std::is_same_v<
        typename DefaultParams::HashGenerator,
        highvoronoi::FNV64_128HashGenerator>);
    static_assert(std::is_same_v<
        typename DefaultParams::ContainerMode,
        highvoronoi::DirectHash>);
    static_assert(std::is_same_v<
        typename DefaultRay::Method,
        highvoronoi::CombinedRaycast>);

    auto boundary = unit_square();

    // Level 1: no database, lock, hash or ray-caster type is mentioned.
    auto mesh = highvoronoi::voronoi_mesh<Dimension>(
        initial_points,
        8,
        boundary);

    using Mesh = decltype(mesh);
    static_assert(std::is_same_v<typename Mesh::Scalar, double>);
    static_assert(std::is_same_v<typename Mesh::Index, std::uint32_t>);
    static_assert(std::is_same_v<
        typename Mesh::Config::RaycastMethod,
        highvoronoi::CombinedRaycast>);
    static_assert(std::is_same_v<
        typename Mesh::Database::LockType,
        highvoronoi::EmptyLock>);
    static_assert(std::is_same_v<
        typename Mesh::Config::DatabaseHashContainer,
        highvoronoi::DirectHash>);

    (void)highvoronoi::compute(mesh);
    (void)highvoronoi::refine(mesh, added_points, 2);
    (void)highvoronoi::remove(mesh, {std::uint32_t{1}});

    // Level 1 parallel: mesh-parallel construction, StaticHash<16> storage.
    auto parallel_mesh = highvoronoi::voronoi_mesh<Dimension>(
        initial_points,
        8,
        boundary,
        highvoronoi::MultiThread{2});

    using ParallelMesh = decltype(parallel_mesh);
    static_assert(std::is_same_v<
        typename ParallelMesh::Database::LockType,
        highvoronoi::ReadWriteLock>);
    static_assert(std::is_same_v<
        typename ParallelMesh::Config::DatabaseHashContainer,
        highvoronoi::StaticHash<16>>);
    static_assert(std::is_same_v<
        typename ParallelMesh::Config::EdgeHashContainer,
        highvoronoi::StaticHash<16>>);

    (void)highvoronoi::compute(parallel_mesh, 2);

    // Level 2: select an alternate ray caster explicitly while the plumbing
    // remains hidden.
    auto config = highvoronoi::make_voronoi_config<
        highvoronoi::SingleThread,
        highvoronoi::InRangeRaycast>(8, Dimension);
    auto configured_mesh = highvoronoi::voronoi_mesh<Dimension>(
        initial_points,
        8,
        boundary,
        config);
    static_assert(std::is_same_v<
        typename decltype(configured_mesh)::Config::RaycastMethod,
        highvoronoi::InRangeRaycast>);

    // Runtime dimension has a shorthand that takes the dimension explicitly.
    using DynamicBoundary = highvoronoi::Boundary<highvoronoi::Dynamic>;
    DynamicBoundary::Point dynamic_dimensions(2);
    DynamicBoundary::Point dynamic_offset(2);
    dynamic_dimensions << 1.0, 1.0;
    dynamic_offset.setZero();
    auto dynamic_boundary = DynamicBoundary::cuboid(
        dynamic_dimensions,
        dynamic_offset,
        std::vector<std::size_t>{});
    auto dynamic_mesh = highvoronoi::voronoi_mesh<highvoronoi::Dynamic>(
        initial_points,
        8,
        dynamic_boundary,
        2);
    (void)dynamic_mesh;

    // HighVoronoi uses the same three API levels and defaults.
    auto high_mesh = highvoronoi::high_voronoi_mesh<Dimension>(
        initial_points,
        8,
        boundary);
    static_assert(std::is_same_v<typename decltype(high_mesh)::Index, std::uint32_t>);
    static_assert(std::is_same_v<
        typename decltype(high_mesh)::Config::RaycastMethod,
        highvoronoi::CombinedRaycast>);

    (void)highvoronoi::compute(high_mesh);
    (void)highvoronoi::refine(high_mesh, added_points, 2);
    (void)highvoronoi::remove(high_mesh, {std::uint32_t{1}});

    auto dynamic_high_mesh =
        highvoronoi::high_voronoi_mesh<highvoronoi::Dynamic>(
            initial_points,
            8,
            dynamic_boundary,
            2);
    (void)dynamic_high_mesh;

    // Integration Level 1: persistent geometry owner + FastPolygon default.
    auto integration_mesh = highvoronoi::voronoi_mesh<Dimension>(
        initial_points,
        8,
        boundary);
    (void)highvoronoi::compute(integration_mesh);

    auto integral = highvoronoi::voronoi_integral(integration_mesh);
    const auto integration_report = highvoronoi::integrate(integral);
    if (integration_report.updated_cells == 0) {
        std::cerr << "integration facade did not update any cells\n";
        return 2;
    }
    decltype(integral)::CellData geometry_cell;
    if (!integral.read_cell(std::uint32_t{0}, geometry_cell) ||
        !(geometry_cell.volume() > 0.0)) {
        std::cerr << "integration facade did not publish finite geometry\n";
        return 3;
    }

    // A second call on an unchanged mesh must reuse the persistent dirty state.
    const auto clean_report = highvoronoi::integrate(integral);
    if (clean_report.updated_cells != 0) {
        std::cerr << "clean integration facade unexpectedly recomputed cells\n";
        return 4;
    }

    // Level 1 with one scalar function enables one bulk/interface component.
    auto function_integral = highvoronoi::voronoi_integral(
        integration_mesh,
        [](const auto& point) { return point.squaredNorm(); });
    (void)highvoronoi::integrate(function_integral);
    decltype(function_integral)::CellData function_cell;
    if (!function_integral.read_cell(std::uint32_t{0}, function_cell) ||
        function_cell.bulk_integral().size() != 1 ||
        function_cell.interface_integral().size() !=
            function_cell.neighbours().size()) {
        std::cerr << "scalar function integration facade has wrong layout\n";
        return 5;
    }

    // Integration Level 2: explicit persistent storage + algorithm choice.
    highvoronoi::IntegrationConfig integration_config;
    auto polygon_integral = highvoronoi::voronoi_integral(
        integration_mesh,
        integration_config);
    const auto polygon_report = highvoronoi::integrate(
        polygon_integral,
        highvoronoi::Polygon{});
    if (polygon_report.updated_cells == 0) {
        std::cerr << "Level-2 Polygon facade did not update any cells\n";
        return 6;
    }

    // The same Level-1 integration facade works on HighVoronoi wrappers.
    auto integration_high_mesh = highvoronoi::high_voronoi_mesh<Dimension>(
        initial_points,
        8,
        boundary);
    (void)highvoronoi::compute(integration_high_mesh);
    auto high_integral = highvoronoi::voronoi_integral(integration_high_mesh);
    const auto high_integration_report = highvoronoi::integrate(high_integral);
    if (high_integration_report.updated_cells == 0) {
        std::cerr << "HighVoronoi integration facade did not update any cells\n";
        return 7;
    }

    std::cout << "Level-1/2 convenience API smoke test passed\n";
    return 0;
}
